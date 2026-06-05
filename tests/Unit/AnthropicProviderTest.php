<?php

declare(strict_types=1);

namespace LLMesh\Anthropic\Tests\Unit;

use LLMesh\Anthropic\AnthropicProvider;
use LLMesh\Core\Data\Message;
use LLMesh\Core\Data\ToolCall;
use LLMesh\Core\Exceptions\HttpException;
use LLMesh\Core\Exceptions\ProviderException;
use LLMesh\Core\Exceptions\RateLimitException;
use LLMesh\Core\Exceptions\TokenLimitException;
use LLMesh\Core\Exceptions\ValidationException;
use LLMesh\Core\Http\HttpClient;
use PHPUnit\Framework\TestCase;

/**
 * Unit tests for AnthropicProvider.
 *
 * The HttpClient is mocked so no real network calls are made.
 *
 * @covers \LLMesh\Anthropic\AnthropicProvider
 */
final class AnthropicProviderTest extends TestCase
{
    private AnthropicProvider $provider;
    /** @var HttpClient&\PHPUnit\Framework\MockObject\MockObject */
    private HttpClient $mockHttpClient;

    protected function setUp(): void
    {
        $this->mockHttpClient = $this->createStub(HttpClient::class);
        $this->provider = new AnthropicProvider(
            apiKey:    'test-api-key-abc123',
            model:     'claude-sonnet-4-5',
            httpClient: $this->mockHttpClient,
        );
    }

    // -------------------------------------------------------------------------
    // Auth header: x-api-key (NOT Authorization: Bearer)
    // -------------------------------------------------------------------------

    public function testUsesXApiKeyHeaderNotBearer(): void
    {
        $this->mockHttpClient = $this->createMock(HttpClient::class);
        $this->provider = new AnthropicProvider(
            apiKey:    'test-api-key-abc123',
            model:     'claude-sonnet-4-5',
            httpClient: $this->mockHttpClient,
        );
        $this->mockHttpClient
            ->expects($this->once())
            ->method('post')
            ->with(
                $this->isString(),
                $this->isArray(),
                $this->callback(function (array $headers): bool {
                    // Must have x-api-key header
                    $this->assertArrayHasKey('x-api-key', $headers, 'x-api-key header missing');
                    $this->assertSame('test-api-key-abc123', $headers['x-api-key']);

                    // Must NOT have an Authorization header
                    $this->assertArrayNotHasKey('Authorization', $headers, 'Authorization header must not be set');

                    return true;
                }),
            )
            ->willReturn($this->buildChatResponse('Hello!'));

        $this->provider->chat([Message::user('Hi')]);
    }

    public function testIncludesAnthropicVersionHeader(): void
    {
        $this->mockHttpClient = $this->createMock(HttpClient::class);
        $this->provider = new AnthropicProvider(
            apiKey:    'test-api-key-abc123',
            model:     'claude-sonnet-4-5',
            httpClient: $this->mockHttpClient,
        );
        $this->mockHttpClient
            ->expects($this->once())
            ->method('post')
            ->with(
                $this->isString(),
                $this->isArray(),
                $this->callback(function (array $headers): bool {
                    $this->assertArrayHasKey('anthropic-version', $headers);
                    $this->assertSame('2023-06-01', $headers['anthropic-version']);
                    return true;
                }),
            )
            ->willReturn($this->buildChatResponse('Hi'));

        $this->provider->chat([Message::user('Hi')]);
    }

    // -------------------------------------------------------------------------
    // System prompt: extracted as top-level field
    // -------------------------------------------------------------------------

    public function testSystemPromptSentAsTopLevelFieldNotInsideMessages(): void
    {
        $this->mockHttpClient = $this->createMock(HttpClient::class);
        $this->provider = new AnthropicProvider(
            apiKey:    'test-api-key-abc123',
            model:     'claude-sonnet-4-5',
            httpClient: $this->mockHttpClient,
        );
        $this->mockHttpClient
            ->expects($this->once())
            ->method('post')
            ->with(
                $this->isString(),
                $this->callback(function (array $payload): bool {
                    // system must be a top-level key in the payload
                    $this->assertArrayHasKey('system', $payload, 'system must be a top-level payload field');
                    $this->assertSame('You are a helpful assistant.', $payload['system']);

                    // system must NOT appear inside the messages array
                    $roles = array_column($payload['messages'], 'role');
                    $this->assertNotContains('system', $roles, 'system role must not appear in messages array');

                    return true;
                }),
                $this->isArray(),
            )
            ->willReturn($this->buildChatResponse('I am here to help.'));

        $this->provider->chat([
            Message::system('You are a helpful assistant.'),
            Message::user('Hello!'),
        ]);
    }

    // -------------------------------------------------------------------------
    // Non-alternating messages: ValidationException
    // -------------------------------------------------------------------------

    public function testThrowsValidationExceptionOnNonAlternatingMessages(): void
    {
        // ValidationException is thrown by MessageMapper::map() before the
        // HTTP client is ever reached, so no mock interactions happen.
        $this->expectException(ValidationException::class);

        $this->provider->chat([
            Message::user('First'),
            Message::user('Second consecutive user message'),
        ]);
    }

    // -------------------------------------------------------------------------
    // embed(): BadMethodCallException
    // -------------------------------------------------------------------------

    public function testEmbedThrowsBadMethodCallException(): void
    {
        $this->expectException(\BadMethodCallException::class);
        $this->expectExceptionMessage('Anthropic does not support embeddings');

        $this->provider->embed('some text');
    }

    public function testEmbedWithArrayThrowsBadMethodCallException(): void
    {
        $this->expectException(\BadMethodCallException::class);

        $this->provider->embed(['text one', 'text two']);
    }

    // -------------------------------------------------------------------------
    // supports()
    // -------------------------------------------------------------------------

    public function testSupportsStreamingAndTools(): void
    {
        self::assertTrue($this->provider->supports('streaming'));
        self::assertTrue($this->provider->supports('tools'));
    }

    public function testDoesNotSupportEmbeddings(): void
    {
        self::assertFalse($this->provider->supports('embeddings'));
    }

    public function testDoesNotSupportUnknownCapability(): void
    {
        self::assertFalse($this->provider->supports('vision'));
    }

    // -------------------------------------------------------------------------
    // chat(): successful response parsing
    // -------------------------------------------------------------------------

    public function testChatReturnsParsedTextResponse(): void
    {
        $this->mockHttpClient
            ->method('post')
            ->willReturn($this->buildChatResponse('The answer is 42.'));

        $response = $this->provider->chat([Message::user('What is the answer?')]);

        self::assertSame('The answer is 42.', $response->getText());
        self::assertSame('end_turn', $response->getFinishReason());
        self::assertSame(15, $response->getUsage()->getInputTokens());
        self::assertSame(8,  $response->getUsage()->getOutputTokens());
    }

    public function testChatResponseWithToolUseBlockHasToolUseFinishReason(): void
    {
        $raw = [
            'id'         => 'msg_01',
            'type'       => 'message',
            'role'       => 'assistant',
            'stop_reason' => 'tool_use',
            'content'    => [
                [
                    'type'  => 'tool_use',
                    'id'    => 'toolu_01',
                    'name'  => 'get_weather',
                    'input' => ['city' => 'London'],
                ],
            ],
            'usage' => ['input_tokens' => 20, 'output_tokens' => 10],
        ];

        $this->mockHttpClient->method('post')->willReturn($raw);

        $response = $this->provider->chat([Message::user('Weather in London?')]);

        self::assertSame('tool_use', $response->getFinishReason());
    }

    // -------------------------------------------------------------------------
    // parseToolUseBlock(): ToolCall DTO conversion
    // -------------------------------------------------------------------------

    public function testParseToolUseBlockCreatesCorrectToolCallDto(): void
    {
        $block = [
            'type'  => 'tool_use',
            'id'    => 'toolu_xyz_789',
            'name'  => 'search_web',
            'input' => ['query' => 'LLMesh PHP'],
        ];

        $toolCall = AnthropicProvider::parseToolUseBlock($block);

        self::assertInstanceOf(ToolCall::class, $toolCall);
        self::assertSame('toolu_xyz_789', $toolCall->id);
        self::assertSame('search_web',    $toolCall->name);
        self::assertSame(['query' => 'LLMesh PHP'], $toolCall->arguments);
    }

    // -------------------------------------------------------------------------
    // stream(): ChunkDelta from text_delta events
    // -------------------------------------------------------------------------

    public function testStreamYieldsChunkDeltaWithTextOnTextDeltaEvent(): void
    {
        $sseLines = [
            'event: content_block_delta',
            'data: ' . json_encode([
                'type'  => 'content_block_delta',
                'index' => 0,
                'delta' => ['type' => 'text_delta', 'text' => 'Hello, '],
            ]),
            '',
            'event: content_block_delta',
            'data: ' . json_encode([
                'type'  => 'content_block_delta',
                'index' => 0,
                'delta' => ['type' => 'text_delta', 'text' => 'world!'],
            ]),
            '',
            'event: message_delta',
            'data: ' . json_encode([
                'type'  => 'message_delta',
                'delta' => ['stop_reason' => 'end_turn'],
            ]),
            '',
        ];

        // Need a real mock (not stub) to use expects(); stream tests own their mock.
        $mock = $this->createMock(HttpClient::class);
        $mock->expects($this->once())
            ->method('stream')
            ->willReturn((static function () use ($sseLines): \Generator {
                yield from $sseLines;
            })());

        $provider = new AnthropicProvider(
            apiKey:     'test-api-key-abc123',
            model:      'claude-sonnet-4-5',
            httpClient: $mock,
        );

        $stream = $provider->stream([Message::user('Say hello')]);

        $chunks = [];
        foreach ($stream as $chunk) {
            $chunks[] = $chunk;
        }

        // We should have 2 text chunks + 1 finish chunk
        $textChunks = array_filter($chunks, fn ($c) => $c->text !== null);
        $textChunks = array_values($textChunks);

        self::assertCount(2, $textChunks);
        self::assertSame('Hello, ', $textChunks[0]->text);
        self::assertSame('world!',  $textChunks[1]->text);

        $finishChunks = array_filter($chunks, fn ($c) => $c->finishReason !== null);
        $finishChunks = array_values($finishChunks);
        self::assertCount(1, $finishChunks);
        self::assertSame('end_turn', $finishChunks[0]->finishReason);
    }

    public function testStreamSkipsInputJsonDeltaEvents(): void
    {
        $sseLines = [
            'data: ' . json_encode([
                'type'  => 'content_block_delta',
                'index' => 0,
                'delta' => ['type' => 'input_json_delta', 'partial_json' => '{"city":'],
            ]),
            'data: ' . json_encode([
                'type'  => 'message_delta',
                'delta' => ['stop_reason' => 'tool_use'],
            ]),
        ];

        $this->mockHttpClient
            ->method('stream')
            ->willReturn((static function () use ($sseLines): \Generator {
                yield from $sseLines;
            })());

        $stream = $this->provider->stream([Message::user('Weather?')]);

        $chunks = [];
        foreach ($stream as $chunk) {
            $chunks[] = $chunk;
        }

        // Only the finish chunk should be emitted; no text chunk for input_json_delta
        $textChunks = array_filter($chunks, fn ($c) => $c->text !== null);
        self::assertEmpty($textChunks);

        $finishChunks = array_filter($chunks, fn ($c) => $c->finishReason !== null);
        self::assertCount(1, array_values($finishChunks));
    }

    // -------------------------------------------------------------------------
    // HTTP exception mapping
    // -------------------------------------------------------------------------

    public function testThrowsRateLimitExceptionOn429(): void
    {
        $this->mockHttpClient
            ->method('post')
            ->willThrowException(new HttpException(
                'Too many requests',
                429,
                json_encode(['error' => ['retry_after_ms' => 3000]]),
            ));

        $this->expectException(RateLimitException::class);

        $this->provider->chat([Message::user('Hi')]);
    }

    public function testThrowsTokenLimitExceptionOnContextLengthError(): void
    {
        $this->mockHttpClient
            ->method('post')
            ->willThrowException(new HttpException(
                'Bad request',
                400,
                json_encode([
                    'error' => [
                        'type'    => 'invalid_request_error',
                        'message' => 'prompt is too long: context_length exceeded',
                    ],
                ]),
            ));

        $this->expectException(TokenLimitException::class);

        $this->provider->chat([Message::user('Hi')]);
    }

    public function testThrowsProviderExceptionOnGeneric400(): void
    {
        $this->mockHttpClient
            ->method('post')
            ->willThrowException(new HttpException(
                'Bad request',
                400,
                json_encode([
                    'error' => ['type' => 'invalid_request_error', 'message' => 'Bad model name'],
                ]),
            ));

        $this->expectException(ProviderException::class);
        $this->expectExceptionMessage('Bad model name');

        $this->provider->chat([Message::user('Hi')]);
    }

    public function testThrowsProviderExceptionOnServerError(): void
    {
        $this->mockHttpClient
            ->method('post')
            ->willThrowException(new HttpException('Internal Server Error', 500, ''));

        $this->expectException(ProviderException::class);
        $this->expectExceptionMessage('Anthropic server error');

        $this->provider->chat([Message::user('Hi')]);
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    /**
     * Build a minimal valid Anthropic chat response payload.
     *
     * @param  string $text
     * @return array<string, mixed>
     */
    private function buildChatResponse(string $text): array
    {
        return [
            'id'          => 'msg_test_01',
            'type'        => 'message',
            'role'        => 'assistant',
            'stop_reason' => 'end_turn',
            'content'     => [
                ['type' => 'text', 'text' => $text],
            ],
            'usage' => [
                'input_tokens'  => 15,
                'output_tokens' => 8,
            ],
        ];
    }
}
