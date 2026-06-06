<?php

declare(strict_types=1);

namespace LLMesh\Anthropic;

use LLMesh\Core\Contracts\EmbeddingResponseInterface;
use LLMesh\Core\Contracts\ProviderInterface;
use LLMesh\Core\Contracts\ResponseInterface;
use LLMesh\Core\Contracts\StreamInterface;
use LLMesh\Core\Data\ChunkDelta;
use LLMesh\Core\Data\ToolCall;
use LLMesh\Core\Data\ProviderChatResponse;
use LLMesh\Core\Data\ProviderStream;
use LLMesh\Core\Exceptions\HttpException;
use LLMesh\Core\Exceptions\ProviderException;
use LLMesh\Core\Exceptions\RateLimitException;
use LLMesh\Core\Exceptions\TokenLimitException;
use LLMesh\Core\Http\HttpClient;
use LLMesh\Core\Http\HttpClientFactory;

/**
 * Anthropic provider implementing ProviderInterface.
 *
 * Supports Claude models via the Anthropic Messages API.
 */
final class AnthropicProvider implements ProviderInterface
{
    private const API_BASE = 'https://api.anthropic.com/v1';
    private const MESSAGES_ENDPOINT = self::API_BASE . '/messages';

    private array $currentToolCalls = [];

    /**
     * @param string          $apiKey     Anthropic API key (sent via x-api-key header)
     * @param string          $model      Model identifier (default: claude-sonnet-4-5)
     * @param string          $apiVersion Anthropic API version date (default: 2023-06-01)
     * @param HttpClient|null $httpClient Injected HTTP client; auto-created if null
     */
    public function __construct(
        private readonly string $apiKey,
        private readonly string $model = 'claude-sonnet-4-5',
        private readonly string $apiVersion = '2023-06-01',
        private readonly ?HttpClient $httpClient = null,
    ) {
    }

    // -------------------------------------------------------------------------
    // ProviderInterface
    // -------------------------------------------------------------------------

    /**
     * {@inheritdoc}
     */
    public function chat(array $messages, array $options = []): ResponseInterface
    {
        $mapped  = MessageMapper::map($messages);
        $payload = $this->buildPayload($mapped, $options, stream: false);

        try {
            $raw = $this->getHttpClient()->post(
                self::MESSAGES_ENDPOINT,
                $payload,
                $this->getHeaders(),
            );

            $parsed = $this->parseChatResponse($raw);
            return new ProviderChatResponse(
                text: $parsed['text'],
                inputTokens: $parsed['usage']['input_tokens'],
                outputTokens: $parsed['usage']['output_tokens'],
                finishReason: $parsed['finishReason'],
                raw: $raw
            );
        } catch (HttpException $e) {
            $this->handleHttpException($e);
        }
    }

    /**
     * {@inheritdoc}
     *
     * Yields ChunkDelta objects from Anthropic Server-Sent Events.
     */
    public function stream(array $messages, array $options = []): StreamInterface
    {
        $mapped  = MessageMapper::map($messages);
        $payload = $this->buildPayload($mapped, $options, stream: true);

        $this->currentToolCalls = [];

        $generator = (function () use ($payload): \Generator {
            try {
                $lines = $this->getHttpClient()->stream(
                    self::MESSAGES_ENDPOINT,
                    $payload,
                    $this->getHeaders(),
                );

                foreach ($lines as $line) {
                    // SSE data lines start with "data: "
                    if (!str_starts_with($line, 'data: ')) {
                        continue;
                    }

                    $json = substr($line, 6);

                    // [DONE] sentinel
                    if ($json === '[DONE]') {
                        break;
                    }

                    $data = json_decode($json, associative: true);
                    if (!is_array($data)) {
                        continue;
                    }

                    $chunk = $this->parseStreamEvent($data);
                    if ($chunk !== null) {
                        yield $chunk;
                    }
                }
            } catch (HttpException $e) {
                $this->handleHttpException($e);
            }
        })();

        return new ProviderStream($generator);
    }

    /**
     * Anthropic does not provide an embedding API.
     *
     * {@inheritdoc}
     *
     * @throws \BadMethodCallException Always.
     */
    public function embed(string|array $input, array $options = []): EmbeddingResponseInterface
    {
        throw new \BadMethodCallException(
            'Anthropic does not support embeddings. Use a dedicated embedding provider instead.'
        );
    }

    /**
     * {@inheritdoc}
     *
     * Supported capabilities: 'streaming', 'tools'.
     * Unsupported: 'embeddings'.
     */
    public function supports(string $capability): bool
    {
        return match ($capability) {
            'streaming', 'tools' => true,
            default              => false,
        };
    }

    // -------------------------------------------------------------------------
    // Payload construction
    // -------------------------------------------------------------------------

    /**
     * Build the request payload for both chat and streaming endpoints.
     *
     * @param  array{messages: array<int, mixed>, system?: string} $mapped Mapped message data
     * @param  array<string, mixed>                                $options
     * @param  bool                                                $stream
     * @return array<string, mixed>
     */
    private function buildPayload(array $mapped, array $options, bool $stream): array
    {
        $payload = [
            'model'      => $this->model,
            'max_tokens' => $options['max_tokens'] ?? 1024,
            'messages'   => $mapped['messages'],
        ];

        // System prompt is a top-level field, never inside messages
        if (isset($mapped['system'])) {
            $payload['system'] = $mapped['system'];
        }

        if ($stream) {
            $payload['stream'] = true;
        }

        if (isset($options['temperature'])) {
            $payload['temperature'] = $options['temperature'];
        }

        if (isset($options['stop_sequences'])) {
            $payload['stop_sequences'] = $options['stop_sequences'];
        }

        if (!empty($options['tools'])) {
            $payload['tools'] = MessageMapper::formatTools($options['tools']);
        }

        return $payload;
    }

    // -------------------------------------------------------------------------
    // Response parsing
    // -------------------------------------------------------------------------

    /**
     * Map Anthropic stop reasons to canonical values.
     */
    private function mapStopReason(?string $reason): string
    {
        return match ($reason) {
            'end_turn', 'stop_sequence' => 'stop',
            'max_tokens' => 'length',
            'tool_use' => 'tool_calls',
            default => $reason ?? 'stop',
        };
    }

    /**
     * Parse an Anthropic chat completion response.
     *
     * Content blocks may be of type 'text' or 'tool_use'. Text blocks are
     * concatenated; a tool_use block signals finish_reason = 'tool_use'.
     *
     * @param  array<string, mixed> $raw
     * @return array{text: string, usage: array<string, int>, finishReason: string}
     */
    private function parseChatResponse(array $raw): array
    {
        $text         = '';
        $finishReason = $raw['stop_reason'] ?? 'end_turn';

        foreach ($raw['content'] ?? [] as $block) {
            if (($block['type'] ?? '') === 'text') {
                $text .= $block['text'] ?? '';
            } elseif (($block['type'] ?? '') === 'tool_use') {
                // Map Anthropic stop reason to the unified 'tool_use' signal
                $finishReason = 'tool_use';
            }
        }

        return [
            'text'         => $text,
            'usage'        => [
                'input_tokens'  => $raw['usage']['input_tokens'] ?? 0,
                'output_tokens' => $raw['usage']['output_tokens'] ?? 0,
            ],
            'finishReason' => $this->mapStopReason($finishReason),
        ];
    }

    /**
     * Parse a single Anthropic SSE event into a ChunkDelta (or null to skip).
     *
     * @param  array<string, mixed> $event Decoded SSE event payload
     * @return ChunkDelta|null
     */
    private function parseStreamEvent(array $event): ?ChunkDelta
    {
        $type = $event['type'] ?? null;

        if ($type === 'content_block_start') {
            $block = $event['content_block'] ?? [];
            if (($block['type'] ?? '') === 'tool_use') {
                $index = $event['index'] ?? 0;
                $this->currentToolCalls[$index] = [
                    'id' => $block['id'] ?? '',
                    'name' => $block['name'] ?? '',
                    'input' => ''
                ];
            }
            return null;
        }

        if ($type === 'content_block_delta') {
            $delta     = $event['delta'] ?? [];
            $deltaType = $delta['type'] ?? null;

            if ($deltaType === 'text_delta') {
                return ChunkDelta::text($delta['text'] ?? '');
            }

            if ($deltaType === 'input_json_delta') {
                $index = $event['index'] ?? 0;
                if (isset($this->currentToolCalls[$index])) {
                    $this->currentToolCalls[$index]['input'] .= $delta['partial_json'] ?? '';
                }
            }
            return null;
        }

        if ($type === 'content_block_stop') {
            $index = $event['index'] ?? 0;
            if (isset($this->currentToolCalls[$index])) {
                $toolCallData = $this->currentToolCalls[$index];
                unset($this->currentToolCalls[$index]);

                $args = json_decode($toolCallData['input'], true) ?? [];
                
                $toolCall = new ToolCall(
                    id:        $toolCallData['id'],
                    name:      $toolCallData['name'],
                    arguments: $args,
                );

                return ChunkDelta::toolCall($toolCall);
            }
            return null;
        }

        if ($type === 'message_delta') {
            $stopReason = $event['delta']['stop_reason'] ?? null;
            if ($stopReason !== null) {
                return ChunkDelta::finish($this->mapStopReason($stopReason));
            }
            return null;
        }

        return null;
    }

    /**
     * Parse a completed tool_use content block into a ToolCall DTO.
     *
     * @param  array<string, mixed> $block A content block with type='tool_use'
     * @return ToolCall
     */
    public static function parseToolUseBlock(array $block): ToolCall
    {
        return new ToolCall(
            id:        $block['id']    ?? '',
            name:      $block['name']  ?? '',
            arguments: $block['input'] ?? [],
        );
    }

    // -------------------------------------------------------------------------
    // HTTP helpers
    // -------------------------------------------------------------------------

    /**
     * Build the required Anthropic API request headers.
     *
     * @return array<string, string>
     */
    private function getHeaders(): array
    {
        return [
            'x-api-key'         => $this->apiKey,
            'anthropic-version' => $this->apiVersion,
            'Content-Type'      => 'application/json',
        ];
    }

    /**
     * Return the injected HttpClient or create one via the factory.
     */
    private function getHttpClient(): HttpClient
    {
        return $this->httpClient ?? HttpClientFactory::make();
    }

    /**
     * Map Anthropic HTTP errors to LLMesh domain exceptions.
     *
     * @param  HttpException $e
     * @throws RateLimitException  On HTTP 429
     * @throws TokenLimitException On HTTP 400 context-length error
     * @throws ProviderException   On all other errors
     * @return never
     */
    private function handleHttpException(HttpException $e): never
    {
        $statusCode = $e->statusCode();
        $bodyString = $e->responseBody();

        // Decode the JSON body (may be empty or malformed)
        $body = json_decode($bodyString, associative: true) ?? [];

        if ($statusCode === 429) {
            $retryAfterMs = $body['error']['retry_after_ms'] ?? 1000;
            throw new RateLimitException(
                message:    'Rate limit exceeded',
                provider:   'anthropic',
                retryAfter: (int) ceil($retryAfterMs / 1000),
            );
        }

        if ($statusCode === 400) {
            $errorType    = $body['error']['type']    ?? '';
            $errorMessage = $body['error']['message'] ?? 'Bad request';

            if ($errorType === 'invalid_request_error'
                && str_contains($errorMessage, 'context_length')) {
                throw new TokenLimitException(
                    message:  'Context length exceeded',
                    provider: 'anthropic',
                    limit:    200000,
                    used:     200000,
                );
            }

            throw new ProviderException(
                message:  $errorMessage,
                provider: 'anthropic',
            );
        }

        if ($statusCode >= 500) {
            throw new ProviderException(
                message:  'Anthropic server error',
                provider: 'anthropic',
            );
        }

        throw new ProviderException(
            message:  "HTTP error: {$statusCode}",
            provider: 'anthropic',
        );
    }
}
