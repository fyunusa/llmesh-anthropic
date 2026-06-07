<?php

/**
 * LLMesh Demo 06: Observability, Logging & Cost Tracking
 *
 * This script demonstrates how to configure middleware stacks around your provider
 * to log requests (PSR-3 compliant) and automatically track API token usage and cost.
 */

// Load the bootstrap file to handle autoloading and API key setup.
$apiKey = require __DIR__ . '/bootstrap.php';

use LLMesh\Core\LLMesh;
use LLMesh\Core\Generators\GenerateTextOptions;
use LLMesh\Core\Observability\MiddlewareStack;
use LLMesh\Core\Observability\LoggingMiddleware;
use LLMesh\Core\Observability\CostTrackingMiddleware;
use LLMesh\Core\Observability\UsageTracker;
use LLMesh\Core\Observability\CostCalculator;
use LLMesh\Anthropic\AnthropicProvider;
use Psr\Log\AbstractLogger;

// 1. Register Claude Pricing
// We register pricing for common Claude model tags in the CostCalculator.
// Format: model, input_cost_per_1M_tokens, output_cost_per_1M_tokens (in USD)
CostCalculator::setPricing('claude-sonnet-4-5-20250929', 3.00, 15.00);
CostCalculator::setPricing('claude-3-5-sonnet-20241022', 3.00, 15.00);

// 2. Simple PSR-3 Console Logger
// Middlewares accept any PSR-3 logger (Monolog, Bugsnag, etc.).
// Here we write a simple logger that prints structured data directly to the console.
class ConsoleLogger extends AbstractLogger
{
    public function log($level, string|\Stringable $message, array $context = []): void
    {
        $time = date('Y-m-d H:i:s');
        $levelUpper = strtoupper((string)$level);
        echo "\n📝 [Logger - {$levelUpper}] {$message}\n";
        if (!empty($context)) {
            echo "   Context: " . json_encode($context, JSON_PRETTY_PRINT) . "\n";
        }
        echo "------------------------------------------------\n";
    }
}

echo "=== LLMesh Observability & Cost Tracking Demo ===\n\n";

try {
    $rawProvider = new AnthropicProvider($apiKey);
    
    // 3. Set Up Tracker and Logger
    $tracker = new UsageTracker(); // Accumulates tokens/costs across multiple requests
    $logger = new ConsoleLogger();

    // 4. Wrap the Provider using MiddlewareStack
    // Middleware wraps the provider like an onion.
    // Order: CostTracking wraps Logging, which wraps rawProvider.
    $provider = MiddlewareStack::wrap($rawProvider)
        ->with(new LoggingMiddleware($logger))
        ->with(new CostTrackingMiddleware($tracker));

    // Call 1: Basic text generation
    echo "Sending request 1 (generateText)...\n";
    $response = LLMesh::generateText(
        $provider,
        GenerateTextOptions::make()->withPrompt('Say hello in German.')
    );
    echo "Response: " . trim($response->getText()) . "\n";

    // Call 2: Streaming generation (Middleware also hooks into streams!)
    echo "\nSending request 2 (streamText)...\n";
    $stream = LLMesh::streamText(
        $provider,
        GenerateTextOptions::make()->withPrompt('Say hello in French.')
    );
    
    echo "Stream Output: ";
    foreach ($stream as $chunk) {
        echo $chunk->text;
        flush();
    }
    echo "\n";

    // 5. Print the Accumulated Cost & Usage
    $summary = $tracker->getSummary();
    echo "\n=== Accumulated Usage & Cost Summary ===\n";
    echo " - Total API Calls: " . $summary['calls'] . "\n";
    echo " - Input Tokens:    " . $summary['tokens_in'] . "\n";
    echo " - Output Tokens:   " . $summary['tokens_out'] . "\n";
    echo " - Total Tokens:    " . $summary['total_tokens'] . "\n";
    echo " - Estimated Cost:  $" . number_format($summary['cost_usd'], 6) . " USD\n";
    echo "========================================\n";

} catch (\Throwable $e) {
    echo "❌ Error occurred: " . $e->getMessage() . "\n";
}
