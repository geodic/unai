# unia

[![Crates.io](https://img.shields.io/crates/v/unia.svg)](https://crates.io/crates/unia)
[![Documentation](https://docs.rs/unia/badge.svg)](https://docs.rs/unia)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> **⚠️ Warning: Heavy Development**
> This library is currently under active and heavy development. APIs are subject to change, and future updates may introduce breaking changes. Use with caution in production environments.

**unia** is a pragmatic, provider-agnostic Rust library designed to unify interactions with various Large Language Model (LLM) providers. It abstracts away the differences between APIs (OpenAI, Anthropic, Gemini, etc.) into a single, consistent interface, while providing powerful features like automatic tool execution (Agents) and Model Context Protocol (MCP) integration.

If you find `unia` helpful, please consider leaving a ⭐, it helps us a lot!

## Key Features

### Universal Client Interface
Write your code once and switch providers with a single line of configuration. `unia` normalizes:
- **Authentication**: Standardized API key handling.
- **Request/Response Models**: Unified `Message`, `Part`, and `Response` structs.
- **Streaming**: Consistent Server-Sent Events (SSE) handling across all providers.

### Agentic Workflow
The `Agent` struct wraps any `Client` to provide an autonomous loop:
- **Automatic Tool Execution**: The agent handles the "LLM calls tool -> Execute tool -> Send result back" loop automatically.
- **Iteration Control**: Configurable maximum iterations to prevent infinite loops.
- **State Management**: Maintains conversation history during the execution loop.

### Model Context Protocol (MCP) Support
Built-in support for the [Model Context Protocol](https://modelcontextprotocol.io/):
- **Tool Integration**: Seamlessly use MCP servers to provide tools to your agents.
- **Resource Access**: Access and read resources directly from MCP servers.
- **Prompt Support**: List and retrieve prompts from MCP servers for dynamic template usage.

## Supported Providers

Here is an exhaustive list of all the providers we currently support (more to come):

- OpenAI (e.g., GPT-5, o3)
- Anthropic (e.g., Claude 4.5 Sonnet, Opus)
- Google Gemini (e.g., Gemini 3.0 Flash, Pro)
- Groq (e.g., Grok)
- Mistral (e.g., Mistral Large)
- DeepSeek
- Perplexity
- OpenRouter
- Together
- Fireworks
- Hyperbolic
- Moonshot
- xAI
- Ollama (local models)

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
unia = "0.1.0"
tokio = { version = "1.0", features = ["full"] }
```

## Simple Example

```rust
use unia::client::Client;
use unia::model::{Message, Part};
use unia::providers::{OpenAI, Provider};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("OPENAI_API_KEY")?;

    // Create client using the factory
    let client = OpenAI::create(api_key, "gpt-5".to_string());
    
    // Create a message
    let messages = vec![
        Message::User(vec![
            Part::Text {
                content: "Hello!".to_string(),
                finished: true,
            }
        ])
    ];

    // Send request
    let response = client.request(messages, vec![]).await?;
    
    // Print response content
    if let Some(msg) = response.data.first() {
        if let Some(content) = msg.content() {
            println!("Response: {}", content);
        }
    }

    Ok(())
}
```
See `examples/` for more detailed example usages.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
