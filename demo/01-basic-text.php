<?php

/**
 * LLMesh Demo 01: Basic Text Generation
 *
 * This script demonstrates the absolute simplest way to generate text using
 * LLMesh with the Anthropic Claude model. It is designed for beginners.
 */

// Load the bootstrap file to handle autoloading and API key setup.
$apiKey = require __DIR__ . '/bootstrap.php';

use LLMesh\Core\LLMesh;
use LLMesh\Core\Generators\GenerateTextOptions;
use LLMesh\Anthropic\AnthropicProvider;

echo "=== LLMesh Basic Text Generation Demo ===\n\n";

try {
    // 1. Initialize the Provider
    // The AnthropicProvider handles communication with the Claude API.
    // By default, it uses 'claude-sonnet-4-5'.
    echo "Initializing Anthropic Provider...\n";
    $provider = new AnthropicProvider($apiKey);

    // 2. Configure Options
    // We use GenerateTextOptions to define our prompt and other LLM parameters.
    $prompt = 'Why is the sky blue? Answer in exactly one sentence.';
    echo "Prompt: \"{$prompt}\"\n\n";

    $options = GenerateTextOptions::make()
        ->withPrompt($prompt)
        ->withTemperature(0.7); // Creative temperature (0.0 = deterministic, 1.0 = highly creative)

    // 3. Generate Text
    // We pass the provider and the options to the LLMesh facade.
    echo "Sending request to Claude...\n";
    $response = LLMesh::generateText($provider, $options);

    // 4. Inspect the Response
    // LLMesh returns a TextResponse object containing the text and metadata.
    echo "\n=== Response Details ===\n";
    echo "Generated Text:\n";
    echo "------------------------------------------------\n";
    echo $response->getText() . "\n";
    echo "------------------------------------------------\n\n";

    // Metadata details
    $usage = $response->getUsage();
    echo "Metadata & Token Usage:\n";
    echo " - Finish Reason:   " . $response->getFinishReason() . "\n"; // e.g. 'stop' (finished normally)
    echo " - Input Tokens:    " . $usage->getInputTokens() . "\n";   // Tokens sent in prompt
    echo " - Output Tokens:   " . $usage->getOutputTokens() . "\n";  // Tokens generated in response
    echo " - Total Tokens:    " . $usage->getTotalTokens() . "\n";   // Sum of input + output tokens
    echo "=========================================\n";

} catch (\Throwable $e) {
    echo "❌ Error occurred: " . $e->getMessage() . "\n";
}
