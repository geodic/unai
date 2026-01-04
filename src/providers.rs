//! LLM provider implementations.

use crate::client::Client;
use crate::options::{ModelOptions, TransportOptions};

/// Trait for LLM providers that can create configured clients.
pub trait Provider {
    /// The client type produced by this provider.
    type Client: Client;

    /// Create a new client with the given API key.
    fn create(api_key: String) -> Self::Client;

    /// Create a new client with custom options.
    fn create_with_options(
        api_key: String,
        model_options: ModelOptions<<Self::Client as Client>::ModelProvider>,
        transport_options: TransportOptions,
    ) -> Self::Client;
}

pub mod anthropic;
pub mod deepseek;
pub mod fireworks;
pub mod gemini;
pub mod groq;
pub mod hyperbolic;
pub mod mistral;
pub mod moonshot;
pub mod ollama;
pub mod openai;
pub mod openrouter;
pub mod perplexity;
pub mod together;
pub mod xai;

// Re-export for convenience
pub use anthropic::{Anthropic, AnthropicClient, AnthropicModel};
pub use deepseek::{DeepSeek, DeepSeekClient, DeepSeekModel};
pub use fireworks::{Fireworks, FireworksClient, FireworksModel};
pub use gemini::{Gemini, GeminiClient, GeminiModel};
pub use groq::{Groq, GroqClient, GroqModel};
pub use hyperbolic::{Hyperbolic, HyperbolicClient, HyperbolicModel};
pub use mistral::{Mistral, MistralClient, MistralModel};
pub use moonshot::{Moonshot, MoonshotClient, MoonshotModel};
pub use ollama::{Ollama, OllamaClient, OllamaModel};
pub use openai::{OpenAi, OpenAiClient, OpenAiModel};
pub use openrouter::{OpenRouter, OpenRouterClient, OpenRouterModel};
pub use perplexity::{Perplexity, PerplexityClient, PerplexityModel};
pub use together::{Together, TogetherClient, TogetherModel};
pub use xai::{XAIClient, XAIModel, XAI};
