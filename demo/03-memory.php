<?php

/**
 * LLMesh Demo 03: Conversation Memory
 *
 * This script demonstrates how to make Claude remember context and details
 * across multiple rounds of a chat conversation using LLMesh memory storage.
 */

// Load the bootstrap file to handle autoloading and API key setup.
$apiKey = require __DIR__ . '/bootstrap.php';

use LLMesh\Core\LLMesh;
use LLMesh\Core\Generators\GenerateTextOptions;
use LLMesh\Core\Memory\InMemoryStore;
use LLMesh\Anthropic\AnthropicProvider;

echo "=== LLMesh Conversation Memory Demo ===\n\n";

try {
    $provider = new AnthropicProvider($apiKey);

    // 1. Initialize Memory Store
    // InMemoryStore stores the message history in local PHP array memory.
    // For production, you can swap this with RedisStore or DatabaseStore.
    $memoryStore = new InMemoryStore();

    // 2. Define a Unique Session ID
    // This allows you to differentiate between multiple user chats.
    $sessionId = 'user-session-12345';

    // --- Round 1: Introduction ---
    echo "Round 1: Sending user details to Claude...\n";
    
    $prompt1 = "Hi! My name is Sarah and my favorite color is emerald green.";
    echo "Sarah: \"{$prompt1}\"\n";

    // Pass the memory store and session ID into the options
    $options1 = GenerateTextOptions::make()
        ->withPrompt($prompt1)
        ->withMemory($memoryStore, $sessionId);

    $response1 = LLMesh::generateText($provider, $options1);
    echo "Claude: " . $response1->getText() . "\n\n";

    // --- Round 2: Memory Recall ---
    echo "Round 2: Asking Claude to recall information...\n";
    
    $prompt2 = "What is my name and what is my favorite color?";
    echo "Sarah: \"{$prompt2}\"\n";

    // We MUST pass the same memory store and session ID to access history
    $options2 = GenerateTextOptions::make()
        ->withPrompt($prompt2)
        ->withMemory($memoryStore, $sessionId);

    $response2 = LLMesh::generateText($provider, $options2);
    echo "Claude: " . $response2->getText() . "\n\n";

    echo "Memory test completed successfully! Claude retained context across requests.\n";

} catch (\Throwable $e) {
    echo "❌ Error occurred: " . $e->getMessage() . "\n";
}
