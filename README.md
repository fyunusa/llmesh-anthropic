# LLMesh Anthropic Provider

[![Latest Stable Version](https://poser.pugx.org/llmesh/anthropic/v)](https://packagist.org/packages/llmesh/anthropic)
[![PHP Version](https://img.shields.io/badge/php-%5E8.1-blue.svg)](https://packagist.org/packages/llmesh/anthropic)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An Anthropic provider adapter for the [LLMesh Core](https://github.com/fyunusa/llmesh) framework, allowing you to use Anthropic Claude models seamlessly with a unified API.

---

## Installation

Install via Composer:

```bash
composer require llmesh/anthropic
```

---

## Quick Start

```php
use LLMesh\Anthropic\AnthropicProvider;
use LLMesh\Core\LLMesh;
use LLMesh\Core\Generators\GenerateTextOptions;

$provider = new AnthropicProvider(getenv('ANTHROPIC_API_KEY'));

// Simple Text Generation
$response = LLMesh::generateText(
    $provider,
    GenerateTextOptions::make()->withPrompt('Say hello!')
);

echo $response->getText();
```

---

## Supported Capabilities

- **Chat Completions**: standard, streaming, and tool calls using models like `claude-sonnet-4-5`, `claude-opus-4-5`, and `claude-haiku-3-5`.
- **Structured Outputs**: validated object generation via native tool mode.
