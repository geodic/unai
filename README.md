# unai - Universal AI Client Library

A small, pragmatic Rust library providing a provider-agnostic LLM client architecture with a fully generic options system.

## Features

- **Async-first**: Built on `tokio` and `reqwest`.
- **Provider-agnostic**: Common trait-based design for all providers.
- **Generic Options**: Flexible configuration for models and transport.
- **Streaming Support**: Unified streaming interface via Server-Sent Events (SSE).
- **Type-safe**: Strong typing for requests, responses, and tools.
- **Extensible**: Easy to add new providers.

## Supported Providers

- OpenAI
- Anthropic (Claude)
- Google Gemini
- Groq
- Mistral AI
- DeepSeek
- Perplexity
- OpenRouter
- Together AI
- Fireworks AI
- Hyperbolic
- Moonshot AI
- xAI (Grok)
- Ollama (Local)

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
unai = "0.1.0"
tokio = { version = "1.0", features = ["full"] }
```

## Usage

```rust
use unai::{Agent, Context};
use unai::client::Client;
use unai::model::{Message, Role};
use unai::options::{HttpTransport, ModelOptions, TransportOptions};
use unai::providers::{OpenAiClient, OpenAiModel};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("OPENAI_API_KEY")?;

    // Configure model options
    let model_options = ModelOptions {
        model: Some("gpt-4o".to_string()),
        provider: OpenAiModel::default(),
        ..Default::default()
    };

    // Configure transport options
    let transport_options = TransportOptions::new(HttpTransport::new(api_key));
    
    // Create agent
    let agent = Agent::<OpenAiClient>::new(model_options, transport_options);

    // Send request
    let messages = vec![Message::Text {
        role: Role::User,
        content: "Hello!".to_string(),
    }];

    let response = agent.chat(Context::with_messages(messages)).await?;
    println!("Response: {:?}", response.data.first().and_then(|m| m.content()));

    Ok(())
}
```

See `examples/` for more provider-specific examples.
