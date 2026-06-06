<?php

declare(strict_types=1);

namespace LLMesh\Anthropic;

use LLMesh\Core\Data\Message;
use LLMesh\Core\Data\MessageRole;
use LLMesh\Core\Exceptions\ValidationException;

/**
 * Converts LLMesh Message DTOs to Anthropic API format.
 *
 * Handles:
 * - Extraction of the system prompt as a top-level field (not inside messages)
 * - Validation that only one system message exists
 * - Validation that non-system messages strictly alternate user/assistant
 * - Tool result formatting: role=user, content type=tool_result with tool_use_id
 * - Tool definition formatting: uses input_schema instead of OpenAI's parameters
 */
final class MessageMapper
{
    /**
     * Convert an array of Message DTOs to Anthropic API format.
     *
     * Returns an array with:
     *   - 'messages': the formatted messages array (system excluded)
     *   - 'system' (optional): the extracted system prompt string
     *
     * @param  Message[] $messages
     * @return array{messages: array<int, array<string, mixed>>, system?: string}
     *
     * @throws ValidationException If a non-Message value is found, more than one system
     *                             message exists, the first non-system message is not from
     *                             the user, or consecutive messages share the same role.
     */
    public static function map(array $messages): array
    {
        if (empty($messages)) {
            return ['messages' => []];
        }

        $system = null;
        $mappedMessages = [];

        foreach ($messages as $index => $message) {
            if (!$message instanceof Message) {
                throw new ValidationException(
                    'All messages must be Message instances',
                    errors: [
                        'messages' => "Expected Message DTO at index {$index}",
                    ],
                );
            }

            // System prompt is extracted as a top-level API field, never sent in messages
            if ($message->role === MessageRole::SYSTEM) {
                if ($system !== null) {
                    throw new ValidationException(
                        'Multiple system messages are not allowed',
                        errors: [
                            'messages' => 'Only one system message is permitted per request',
                        ],
                    );
                }
                $system = $message->content;
                continue;
            }

            $mappedMessages[] = self::mapMessage($message);
        }

        // Anthropic requires strict user/assistant alternation
        self::validateAlternation($mappedMessages);

        $result = ['messages' => $mappedMessages];

        if ($system !== null) {
            $result['system'] = $system;
        }

        return $result;
    }

    /**
     * Format tool definitions for the Anthropic API.
     *
     * Anthropic uses `input_schema` instead of OpenAI's `parameters` key.
     * Accepts either objects implementing toArray() or raw associative arrays.
     *
     * @param  array<int, array<string, mixed>|object> $tools
     * @return array<int, array<string, mixed>>
     */
    public static function formatTools(array $tools): array
    {
        return array_map(static function (mixed $tool): array {
            if (is_object($tool) && method_exists($tool, 'toArray')) {
                $toolArray = $tool->toArray();
            } elseif (is_array($tool)) {
                $toolArray = $tool;
            } else {
                throw new \InvalidArgumentException(
                    'Each tool must be an array or an object with a toArray() method'
                );
            }

            if (isset($toolArray['type']) && $toolArray['type'] === 'function' && isset($toolArray['function'])) {
                $toolArray = $toolArray['function'];
            }

            return [
                'name'         => $toolArray['name'],
                'description'  => $toolArray['description'] ?? '',
                // Anthropic uses input_schema; OpenAI uses parameters
                'input_schema' => $toolArray['parameters'] ?? $toolArray['input_schema'] ?? (object) [],
            ];
        }, $tools);
    }

    // -------------------------------------------------------------------------
    // Private helpers
    // -------------------------------------------------------------------------

    /**
     * Convert a single Message DTO to an Anthropic API message array.
     *
     * Tool result messages (role=tool in LLMesh) are mapped to role=user with
     * a structured content block of type tool_result, as required by Anthropic.
     *
     * @param  Message $message
     * @return array<string, mixed>
     */
    private static function mapMessage(Message $message): array
    {
        // Tool result: LLMesh uses role=tool; Anthropic wants role=user + tool_result block
        if ($message->role === MessageRole::TOOL) {
            return [
                'role'    => 'user',
                'content' => [
                    [
                        'type'        => 'tool_result',
                        'tool_use_id' => $message->toolCallId,
                        'content'     => $message->content,
                    ],
                ],
            ];
        }

        if ($message->role === MessageRole::ASSISTANT) {
            $content = $message->content;
            if (str_starts_with($content, '[')) {
                try {
                    $decoded = json_decode($content, true, 512, JSON_THROW_ON_ERROR);
                    if (is_array($decoded) && !empty($decoded) && isset($decoded[0]['id'], $decoded[0]['name'])) {
                        $blocks = [];
                        foreach ($decoded as $tc) {
                            $blocks[] = [
                                'type'  => 'tool_use',
                                'id'    => $tc['id'],
                                'name'  => $tc['name'],
                                'input' => $tc['arguments'] ?? [],
                            ];
                        }
                        return [
                            'role'    => 'assistant',
                            'content' => $blocks,
                        ];
                    }
                } catch (\JsonException) {
                    // Fall back to plain text mapping
                }
            }
        }

        return [
            'role'    => $message->role->value,
            'content' => $message->content,
        ];
    }

    /**
     * Validate that the already-mapped messages strictly alternate user/assistant.
     *
     * Rules:
     *   1. The first message must have role "user".
     *   2. No two consecutive messages may share the same role.
     *
     * @param  array<int, array<string, mixed>> $messages
     *
     * @throws ValidationException
     */
    private static function validateAlternation(array $messages): void
    {
        if (empty($messages)) {
            return;
        }

        $lastRole = null;

        foreach ($messages as $index => $message) {
            $currentRole = $message['role'];

            if ($lastRole === null) {
                if ($currentRole !== 'user') {
                    throw new ValidationException(
                        'The first message must be from the user role',
                        errors: [
                            'messages' => "Message at index {$index} must be 'user', got '{$currentRole}'",
                        ],
                    );
                }
            } elseif ($currentRole === $lastRole) {
                throw new ValidationException(
                    'Messages must alternate between user and assistant roles',
                    errors: [
                        'messages' => "Message at index {$index} breaks alternation: "
                            . "'{$lastRole}' cannot be followed by '{$currentRole}'",
                    ],
                );
            }

            $lastRole = $currentRole;
        }
    }
}
