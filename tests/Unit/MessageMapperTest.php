<?php

declare(strict_types=1);

namespace LLMesh\Anthropic\Tests\Unit;

use LLMesh\Anthropic\MessageMapper;
use LLMesh\Core\Data\Message;
use LLMesh\Core\Exceptions\ValidationException;
use PHPUnit\Framework\TestCase;

/**
 * Tests for MessageMapper in complete isolation from the HTTP layer.
 *
 * @covers \LLMesh\Anthropic\MessageMapper
 */
final class MessageMapperTest extends TestCase
{
    // -------------------------------------------------------------------------
    // System prompt extraction
    // -------------------------------------------------------------------------

    public function testSystemMessageIsExtractedAsTopLevelField(): void
    {
        $messages = [
            Message::system('You are a helpful assistant.'),
            Message::user('Hello!'),
        ];

        $result = MessageMapper::map($messages);

        // System prompt must appear as top-level key
        self::assertSame('You are a helpful assistant.', $result['system']);

        // System message must NOT appear inside the messages array
        $roles = array_column($result['messages'], 'role');
        self::assertNotContains('system', $roles);
    }

    public function testSystemMessageIsNotIncludedInMessagesArray(): void
    {
        $messages = [
            Message::system('Be concise.'),
            Message::user('What is PHP?'),
            Message::assistant('PHP is a scripting language.'),
        ];

        $result = MessageMapper::map($messages);

        self::assertCount(2, $result['messages']);
        self::assertSame('user',      $result['messages'][0]['role']);
        self::assertSame('assistant', $result['messages'][1]['role']);
    }

    public function testNoSystemMessageProducesNoSystemKey(): void
    {
        $messages = [Message::user('Hi')];

        $result = MessageMapper::map($messages);

        self::assertArrayNotHasKey('system', $result);
    }

    public function testMultipleSystemMessagesThrowsValidationException(): void
    {
        $this->expectException(ValidationException::class);
        $this->expectExceptionMessage('Multiple system messages are not allowed');

        MessageMapper::map([
            Message::system('First system message'),
            Message::system('Second system message'),
            Message::user('Hello'),
        ]);
    }

    // -------------------------------------------------------------------------
    // Message alternation validation
    // -------------------------------------------------------------------------

    public function testValidAlternatingMessagesPassesWithoutException(): void
    {
        $messages = [
            Message::user('Hello'),
            Message::assistant('Hi there'),
            Message::user('How are you?'),
            Message::assistant('Great, thanks!'),
        ];

        $result = MessageMapper::map($messages);

        self::assertCount(4, $result['messages']);
    }

    public function testNonAlternatingMessagesThrowsValidationException(): void
    {
        $this->expectException(ValidationException::class);
        $this->expectExceptionMessage('Messages must alternate between user and assistant roles');

        MessageMapper::map([
            Message::user('First user message'),
            Message::user('Second consecutive user message'), // violation
        ]);
    }

    public function testConsecutiveAssistantMessagesThrowsValidationException(): void
    {
        $this->expectException(ValidationException::class);
        $this->expectExceptionMessage('Messages must alternate between user and assistant roles');

        MessageMapper::map([
            Message::user('Hello'),
            Message::assistant('Hi'),
            Message::assistant('I am here'), // violation
        ]);
    }

    public function testFirstMessageNotUserThrowsValidationException(): void
    {
        $this->expectException(ValidationException::class);
        $this->expectExceptionMessage('The first message must be from the user role');

        MessageMapper::map([
            Message::assistant('I speak first'), // violation
        ]);
    }

    public function testValidationExceptionCarriesErrorDetails(): void
    {
        try {
            MessageMapper::map([
                Message::user('A'),
                Message::user('B'),
            ]);
            self::fail('Expected ValidationException was not thrown');
        } catch (ValidationException $e) {
            self::assertArrayHasKey('messages', $e->errors());
        }
    }

    // -------------------------------------------------------------------------
    // Tool result message mapping
    // -------------------------------------------------------------------------

    public function testToolResultMessageMappedToUserRoleWithToolResultBlock(): void
    {
        $messages = [
            Message::user('What is the weather?'),
            Message::assistant(''),                              // tool_use response
            Message::tool('{"temp": 22}', 'toolu_01', 'get_weather'),
        ];

        $result = MessageMapper::map($messages);

        // Tool message must be role=user with structured content block
        $toolMsg = $result['messages'][2];
        self::assertSame('user', $toolMsg['role']);
        self::assertIsArray($toolMsg['content']);
        self::assertSame('tool_result',  $toolMsg['content'][0]['type']);
        self::assertSame('toolu_01',     $toolMsg['content'][0]['tool_use_id']);
        self::assertSame('{"temp": 22}', $toolMsg['content'][0]['content']);
    }

    // -------------------------------------------------------------------------
    // Tool definition formatting (input_schema)
    // -------------------------------------------------------------------------

    public function testFormatToolsUsesInputSchemaKey(): void
    {
        $tools = [
            [
                'name'        => 'get_weather',
                'description' => 'Get current weather',
                'parameters'  => [
                    'type'       => 'object',
                    'properties' => ['city' => ['type' => 'string']],
                ],
            ],
        ];

        $formatted = MessageMapper::formatTools($tools);

        self::assertCount(1, $formatted);
        self::assertArrayHasKey('input_schema', $formatted[0]);
        self::assertArrayNotHasKey('parameters', $formatted[0]);
        self::assertSame('get_weather', $formatted[0]['name']);
        self::assertSame('object', $formatted[0]['input_schema']['type']);
    }

    public function testFormatToolsPassThroughInputSchemaIfAlreadyPresent(): void
    {
        $tools = [
            [
                'name'         => 'search',
                'description'  => 'Search the web',
                'input_schema' => ['type' => 'object', 'properties' => []],
            ],
        ];

        $formatted = MessageMapper::formatTools($tools);

        self::assertSame('object', $formatted[0]['input_schema']['type']);
    }

    // -------------------------------------------------------------------------
    // Edge cases
    // -------------------------------------------------------------------------

    public function testEmptyMessagesArrayReturnsEmptyMessages(): void
    {
        $result = MessageMapper::map([]);

        self::assertSame([], $result['messages']);
        self::assertArrayNotHasKey('system', $result);
    }

    public function testNonMessageValueThrowsValidationException(): void
    {
        $this->expectException(ValidationException::class);
        $this->expectExceptionMessage('All messages must be Message instances');

        /** @phpstan-ignore-next-line */
        MessageMapper::map([['role' => 'user', 'content' => 'raw array']]);
    }
}
