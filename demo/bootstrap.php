<?php

/**
 * Bootstrap file for LLMesh Anthropic Demos.
 *
 * This file handles Composer autoloading and retrieves the ANTHROPIC_API_KEY
 * from the environment or a local .env file.
 */

// 1. Load Composer Autoloader
$autoloader = dirname(__DIR__) . '/vendor/autoload.php';
if (!file_exists($autoloader)) {
    echo "⚠️  Autoload Error: Please run 'composer install' in the repository root first.\n";
    exit(1);
}
require $autoloader;

// 2. Simple helper to parse .env file if vlucas/phpdotenv is not installed
function loadEnvFile(string $path): void
{
    if (!file_exists($path)) {
        return;
    }
    $lines = file($path, FILE_IGNORE_NEW_LINES | FILE_SKIP_EMPTY_LINES);
    foreach ($lines as $line) {
        $line = trim($line);
        if (empty($line) || str_starts_with($line, '#')) {
            continue;
        }
        if (str_contains($line, '=')) {
            list($name, $value) = explode('=', $line, 2);
            $name = trim($name);
            $value = trim($value, " \t\n\r\0\x0B\"'");
            $_ENV[$name] = $value;
            putenv("{$name}={$value}");
        }
    }
}

// Load env files from repo root or the lab directory
loadEnvFile(dirname(__DIR__) . '/.env');
loadEnvFile(dirname(dirname(__DIR__)) . '/llmesh-lab/.env');

$apiKey = isset($_ENV['ANTHROPIC_API_KEY']) ? $_ENV['ANTHROPIC_API_KEY'] : getenv('ANTHROPIC_API_KEY');

if (!$apiKey || empty(trim($apiKey))) {
    echo "⚠️  API Key Error: ANTHROPIC_API_KEY is not set.\n";
    echo "Please set the ANTHROPIC_API_KEY environment variable or create a .env file.\n";
    exit(1);
}

return $apiKey;
