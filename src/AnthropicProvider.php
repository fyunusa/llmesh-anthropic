<?php

declare(strict_types=1);

namespace LLMesh\Anthropic;

use LLMesh\Core\Contracts\EmbeddingResponseInterface;
use LLMesh\Core\Contracts\ProviderInterface;
use LLMesh\Core\Contracts\ResponseInterface;
use LLMesh\Core\Contracts\StreamInterface;
use LLMesh\Core\Data\ChunkDelta;
use LLMesh\Core\Data\ToolCall;
use LLMesh\Core\Exceptions\HttpException;
use LLMesh\Core\Exceptions\ProviderException;
use LLMesh\Core\Exceptions\RateLimitException;
use LLMesh\Core\Exceptions\TokenLimitException;
use LLMesh\Core\Generators\StreamResponse;
use LLMesh\Core\Generators\TextResponse;
use LLMesh\Core\Http\HttpClient;
use LLMesh\Core\Http\HttpClientFactory;

/**
 * Anthropic provider implementing ProviderInterface.
 *
 * Supports Claude models via the Anthropic Messages API.
 *
 * Key differences from OpenAI:
 *   - Auth uses `x-api-key` header, NOT `Authorization: Bearer`
 *   - `anthropic-version` header is required on every request
 *   - System prompt is a top-level API field, not a member of the messages array
 *   - Messages must strictly alternate user/assistant (validated by MessageMapper)
 *   - Tool definitions use `input_schema` instead of `parameters`
 *   - Tool-use responses are content blocks of type `tool_use` with `id`, `name`, `input`
 *   - Tool result messages use role=user with content type `tool_result` + `tool_use_id`
 *   - Streaming events are `content_block_delta` with delta type `text_delta` or `input_json_delta`
 *   - Stop reasons: `end_turn`, `max_tokens`, `stop_sequence`, `tool_use`
 *   - Embeddings are NOT supported; `embed()` throws BadMethodCallException
 */
final class AnthropicProvider implements ProviderInterface
{
    private const API_BASE = 'https://api.anthropic.com/v1';
    private const MESSAGES_ENDPOINT = self::API_BASE . '/messages';

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

            return TextResponse::fromProviderResponse($raw, fn (array $r) => $this->parseChatResponse($r));
        } catch (HttpException $e) {
            $this->handleHttpException($e);
        }
    }

    /**
     * {@inheritdoc}
     *
     * Yields ChunkDelta objects from Anthropic Server-Sent Events.
     * Recognised event types:
     *   - content_block_delta / text_delta      → ChunkDelta::text()
     *   - content_block_delta / input_json_delta → tool argument accumulation (null chunk)
     *   - message_delta (stop_reason present)    → ChunkDelta::finish()
     *   - message_stop                           → stream end sentinel (no chunk emitted)
     */
    public function stream(array $messages, array $options = []): StreamInterface
    {
        $mapped  = MessageMapper::map($messages);
        $payload = $this->buildPayload($mapped, $options, stream: true);

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

                    // [DONE] sentinel (Anthropic sends message_stop event instead, but guard anyway)
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

        return new StreamResponse($generator);
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
            'finishReason' => $finishReason,
        ];
    }

    /**
     * Parse a single Anthropic SSE event into a ChunkDelta (or null to skip).
     *
     * Anthropic SSE event types of interest:
     *   - content_block_delta: carries text_delta or input_json_delta
     *   - message_delta:       carries stop_reason when generation ends
     *   - message_stop:        final sentinel; no data to emit
     *
     * @param  array<string, mixed> $event Decoded SSE event payload
     * @return ChunkDelta|null
     */
    private function parseStreamEvent(array $event): ?ChunkDelta
    {
        $type = $event['type'] ?? null;

        if ($type === 'content_block_delta') {
            $delta     = $event['delta'] ?? [];
            $deltaType = $delta['type'] ?? null;

            if ($deltaType === 'text_delta') {
                return ChunkDelta::text($delta['text'] ?? '');
            }

            // input_json_delta accumulates tool-call arguments; we skip mid-stream
            // — a complete ToolCall DTO is only possible after the full block is assembled.
            return null;
        }

        if ($type === 'content_block_stop') {
            // Signals the end of a content block (tool_use or text).
            // The provider emits a message_delta with stop_reason shortly after.
            return null;
        }

        if ($type === 'message_delta') {
            $stopReason = $event['delta']['stop_reason'] ?? null;
            if ($stopReason !== null) {
                return ChunkDelta::finish($stopReason);
            }
            return null;
        }

        // message_stop, ping, content_block_start, message_start → ignore
        return null;
    }

    /**
     * Parse a completed tool_use content block into a ToolCall DTO.
     *
     * The Anthropic API returns tool_use blocks with:
     *   - id:    unique identifier (matches tool_use_id in subsequent tool_result)
     *   - name:  tool function name
     *   - input: decoded arguments object (already associative array)
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
     * Authentication uses `x-api-key` (NOT `Authorization: Bearer`).
     * The `anthropic-version` header is mandatory.
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
