<?php

/**
 * LLMesh Demo 07: Event Dispatching (PSR-14)
 *
 * This script demonstrates how to configure LLMesh to dispatch lifecycle events.
 * Events are triggered automatically when a generation starts, completes, or fails.
 */

// Load the bootstrap file to handle autoloading and API key setup.
$apiKey = require __DIR__ . '/bootstrap.php';

use LLMesh\Core\LLMesh;
use LLMesh\Core\Generators\GenerateTextOptions;
use LLMesh\Anthropic\AnthropicProvider;
use Psr\EventDispatcher\EventDispatcherInterface;
use LLMesh\Core\Events\GenerationStarted;
use LLMesh\Core\Events\GenerationCompleted;
use LLMesh\Core\Events\GenerationFailed;

// 1. Simple Inline PSR-14 Event Dispatcher
// LLMesh triggers events which can be captured by any framework event listener
// (e.g. Laravel, Symfony). Here is a simple standalone implementation.
class SimpleEventDispatcher implements EventDispatcherInterface
{
    private array $listeners = [];

    // Register a listener callback for a specific event class
    public function addListener(string $eventClass, callable $listener): void
    {
        $this->listeners[$eventClass][] = $listener;
    }

    // Dispatches the event object to all registered listeners
    public function dispatch(object $event): object
    {
        $eventClass = get_class($event);
        if (isset($this->listeners[$eventClass])) {
            foreach ($this->listeners[$eventClass] as $listener) {
                $listener($event);
            }
        }
        return $event;
    }
}

echo "=== LLMesh Event Dispatching Demo ===\n\n";

try {
    $provider = new AnthropicProvider($apiKey);

    // 2. Instantiate and Register Event Listeners
    $dispatcher = new SimpleEventDispatcher();

    // Event 1: GenerationStarted
    // Dispatched immediately before calling the LLM provider
    $dispatcher->addListener(GenerationStarted::class, function (GenerationStarted $event) {
        echo "\n📢 [Event: GenerationStarted]\n";
        echo "   - Provider: " . $event->provider . "\n";
        echo "   - Prompt:   " . ($event->options->prompt ?? '(Messages array passed)') . "\n";
        echo "------------------------------------------------\n";
    });

    // Event 2: GenerationCompleted
    // Dispatched successfully after receiving the LLM response
    $dispatcher->addListener(GenerationCompleted::class, function (GenerationCompleted $event) {
        echo "\n📢 [Event: GenerationCompleted]\n";
        echo "   - Provider:    " . $event->provider . "\n";
        echo "   - Duration:    " . $event->durationMs . " ms\n";
        
        $usage = $event->response->getUsage();
        echo "   - Token Usage: In=" . $usage->getInputTokens() . 
                           ", Out=" . $usage->getOutputTokens() . 
                           ", Total=" . $usage->getTotalTokens() . "\n";
        echo "------------------------------------------------\n";
    });

    // Event 3: GenerationFailed
    // Dispatched if an exception is thrown during generation
    $dispatcher->addListener(GenerationFailed::class, function (GenerationFailed $event) {
        echo "\n📢 [Event: GenerationFailed]\n";
        echo "   - Provider: " . $event->provider . "\n";
        echo "   - Error:    " . $event->exception->getMessage() . "\n";
        echo "------------------------------------------------\n";
    });

    // 3. Inject Dispatcher into the LLMesh Instance
    $llmesh = LLMesh::make()->withEventDispatcher($dispatcher);

    // 4. Run Generation
    echo "Initiating text generation request...\n";
    $response = $llmesh->generateText(
        $provider,
        GenerateTextOptions::make()->withPrompt('Say "PHP is great" in Italian.')
    );

    echo "\nFinal Answer: " . trim($response->getText()) . "\n";

} catch (\Throwable $e) {
    echo "❌ Error occurred: " . $e->getMessage() . "\n";
}
