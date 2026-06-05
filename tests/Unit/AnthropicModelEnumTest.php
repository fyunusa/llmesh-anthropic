<?php

declare(strict_types=1);

namespace LLMesh\Anthropic\Tests\Unit;

use LLMesh\Anthropic\AnthropicModelEnum;
use PHPUnit\Framework\TestCase;

/**
 * @covers \LLMesh\Anthropic\AnthropicModelEnum
 */
final class AnthropicModelEnumTest extends TestCase
{
    // -------------------------------------------------------------------------
    // supportsTools()
    // -------------------------------------------------------------------------

    public function testSonnetSupportsTools(): void
    {
        self::assertTrue(AnthropicModelEnum::CLAUDE_SONNET_45->supportsTools());
    }

    public function testOpusSupportsTools(): void
    {
        self::assertTrue(AnthropicModelEnum::CLAUDE_OPUS_45->supportsTools());
    }

    public function testHaikuSupportsTools(): void
    {
        self::assertTrue(AnthropicModelEnum::CLAUDE_HAIKU_35->supportsTools());
    }

    // -------------------------------------------------------------------------
    // supportsStreaming()
    // -------------------------------------------------------------------------

    public function testSonnetSupportsStreaming(): void
    {
        self::assertTrue(AnthropicModelEnum::CLAUDE_SONNET_45->supportsStreaming());
    }

    public function testOpusSupportsStreaming(): void
    {
        self::assertTrue(AnthropicModelEnum::CLAUDE_OPUS_45->supportsStreaming());
    }

    public function testHaikuSupportsStreaming(): void
    {
        self::assertTrue(AnthropicModelEnum::CLAUDE_HAIKU_35->supportsStreaming());
    }

    // -------------------------------------------------------------------------
    // Enum string values
    // -------------------------------------------------------------------------

    public function testEnumValuesMatchAnthropicModelIdentifiers(): void
    {
        self::assertSame('claude-sonnet-4-5', AnthropicModelEnum::CLAUDE_SONNET_45->value);
        self::assertSame('claude-opus-4-5',   AnthropicModelEnum::CLAUDE_OPUS_45->value);
        self::assertSame('claude-haiku-3-5',  AnthropicModelEnum::CLAUDE_HAIKU_35->value);
    }

    public function testFromValueRoundTrips(): void
    {
        self::assertSame(
            AnthropicModelEnum::CLAUDE_SONNET_45,
            AnthropicModelEnum::from('claude-sonnet-4-5'),
        );
    }
}
