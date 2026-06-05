<?php

declare(strict_types=1);

namespace LLMesh\Anthropic;

/**
 * Anthropic model enumeration with capability metadata.
 *
 * Values use the canonical Anthropic model identifiers.
 */
enum AnthropicModelEnum: string
{
    case CLAUDE_SONNET_45 = 'claude-sonnet-4-5';
    case CLAUDE_OPUS_45   = 'claude-opus-4-5';
    case CLAUDE_HAIKU_35  = 'claude-haiku-3-5';

    /**
     * Check if this model supports tool calling.
     *
     * All current Claude models support tools.
     */
    public function supportsTools(): bool
    {
        return match ($this) {
            self::CLAUDE_SONNET_45,
            self::CLAUDE_OPUS_45,
            self::CLAUDE_HAIKU_35 => true,
        };
    }

    /**
     * Check if this model supports streaming.
     *
     * All current Claude models support streaming.
     */
    public function supportsStreaming(): bool
    {
        return match ($this) {
            self::CLAUDE_SONNET_45,
            self::CLAUDE_OPUS_45,
            self::CLAUDE_HAIKU_35 => true,
        };
    }
}
