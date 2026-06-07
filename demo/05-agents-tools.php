<?php

/**
 * LLMesh Demo 05: Agents & Custom Tools
 *
 * This script demonstrates how to build an Agent that can make decisions and use
 * custom PHP tools/functions to perform tasks (like mathematical calculations).
 */

// Load the bootstrap file to handle autoloading and API key setup.
$apiKey = require __DIR__ . '/bootstrap.php';

use LLMesh\Core\Agents\Agent;
use LLMesh\Core\Tools\Tool;
use LLMesh\Anthropic\AnthropicProvider;

echo "=== LLMesh Agents & Custom Tools Demo ===\n\n";

// 1. Define a Custom PHP Tool
// We build a Tool named 'calculate_shipping_cost'.
// We provide descriptions for the LLM to know when and how to call it.
$shippingCostTool = Tool::make('calculate_shipping_cost')
    ->description('Calculate the shipping cost in USD for a package based on its weight.')
    ->parameters([
        // Define inputs that the LLM must supply
        'weight' => Tool::number('The weight of the package in kilograms (kg)')->required(),
    ])
    ->handler(function (array $params): array {
        // This is the actual PHP code that runs when the LLM decides to call the tool
        $weight = $params['weight'];
        
        // Shipping calculation logic: flat rate $5.00 + $2.50 per kg
        $cost = 5.00 + ($weight * 2.50);
        
        echo "   🔧 [Tool Execution] Running calculate_shipping_cost for weight: {$weight} kg...\n";
        echo "   🔧 [Tool Execution] Calculated cost: \${$cost} USD\n";
        
        // Must return an array (JSON-serializable) back to the LLM
        return [
            'shipping_cost_usd' => $cost,
            'currency'          => 'USD'
        ];
    });

try {
    $provider = new AnthropicProvider($apiKey);

    // 2. Initialize the Agent
    // We pass the provider, system instructions, and list of tools.
    $agent = Agent::make(
        provider:     $provider,
        systemPrompt: 'You are a helpful logistics assistant. Use the calculate_shipping_cost tool when a user asks about shipping costs.',
        tools:        [$shippingCostTool],
        maxSteps:     5 // Limit maximum model turns to avoid infinite loops
    );

    // 3. Register a Step Listener (Optional)
    // We can hook into the step-by-step model execution to print what is happening.
    $agent->onStep(function ($step) {
        echo "🔄 [Agent Step Completed]\n";
        if (!empty($step->toolCalls)) {
            foreach ($step->toolCalls as $call) {
                echo "   👉 Model requested tool: \"{$call->name}\" with arguments: " . json_encode($call->arguments) . "\n";
            }
        } else {
            echo "   👉 Model returned its final answer.\n";
        }
    });

    // 4. Run the Agent
    $query = "I have a package that weighs 12 kilograms. How much will it cost to ship it?";
    echo "User Query: \"{$query}\"\n\n";
    echo "Starting Agent execution...\n";
    echo "------------------------------------------------\n";

    $result = $agent->run($query);

    echo "------------------------------------------------\n";
    echo "\n=== Final Agent Response ===\n";
    echo $result->finalText . "\n";

} catch (\Throwable $e) {
    echo "❌ Error occurred: " . $e->getMessage() . "\n";
}
