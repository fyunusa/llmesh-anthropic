<?php

/**
 * LLMesh Demo 04: Structured JSON Output
 *
 * This script demonstrates how to force the LLM to return data in a specific,
 * structured JSON format matching a predefined schema.
 */

// Load the bootstrap file to handle autoloading and API key setup.
$apiKey = require __DIR__ . '/bootstrap.php';

use LLMesh\Core\LLMesh;
use LLMesh\Core\Generators\GenerateObjectOptions;
use LLMesh\Core\Schema\Schema;
use LLMesh\Anthropic\AnthropicProvider;

echo "=== LLMesh Structured Output Demo ===\n\n";

// 1. Define the Schema
// The Schema class defines the structure we want the LLM to follow.
// It matches standard JSON schema rules.
$schema = Schema::object([
    'name'      => Schema::string()->required()->description('The name of the character'),
    'age'       => Schema::integer()->required()->minimum(0)->description('The age of the character'),
    'class'     => Schema::string()->required()->description('Character class e.g., Wizard, Warrior, Rogue'),
    'abilities' => Schema::array(Schema::string())->required()->description('List of 3 unique special abilities'),
])->required(['name', 'age', 'class', 'abilities']);

try {
    $provider = new AnthropicProvider($apiKey);

    $prompt = 'Create a fantasy RPG character who is a wise old wizard.';
    echo "Prompt: \"{$prompt}\"\n\n";

    // 2. Configure Options
    // We use GenerateObjectOptions and pass the schema
    $options = GenerateObjectOptions::make()
        ->withPrompt($prompt)
        ->withSchema($schema);

    // 3. Generate Object
    // We call generateObject() on the LLMesh facade.
    // It calls the LLM, validates the output, and retries automatically if it's invalid.
    echo "Generating structured object...\n";
    $response = LLMesh::generateObject($provider, $options);

    // 4. Retrieve and Print the parsed array
    // The structured data is available as an associative array on the $response->object property.
    echo "\n=== Parsed RPG Character Data ===\n";
    echo "Successfully parsed to PHP Array:\n";
    print_r($response->object);

    echo "\nAccessing specific fields:\n";
    echo " - Name:      " . $response->object['name'] . "\n";
    echo " - Class:     " . $response->object['class'] . "\n";
    echo " - Abilities: " . implode(', ', $response->object['abilities']) . "\n";

} catch (\LLMesh\Core\Exceptions\ValidationException $e) {
    // This exception is thrown if the model returns JSON that fails schema validation
    echo "❌ Validation Error: The returned response did not match the schema.\n";
    echo "Validation errors:\n";
    print_r($e->errors());
} catch (\Throwable $e) {
    echo "❌ Error occurred: " . $e->getMessage() . "\n";
}
