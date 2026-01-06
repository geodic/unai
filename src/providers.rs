//! LLM provider implementations.

use crate::client::Client;
use crate::options::{ModelOptions, TransportOptions};

/// Trait for LLM providers that can create configured clients.
///
/// This trait acts as a factory for creating `Client` instances.
/// It provides a standard interface for initializing clients with API keys
/// and optional configuration.
pub trait Provider {
    /// The client type produced by this provider.
    type Client: Client;

    /// Create a new client with the given API key and model.
    ///
    /// # Arguments
    /// - `api_key`: The API key for authentication.
    /// - `model`: The model identifier to use.
    fn create(api_key: String, model: String) -> Self::Client;

    /// Create a new client with custom model and transport options.
    ///
    /// # Arguments
    /// - `api_key`: The API key for authentication.
    /// - `model_options`: Configuration for model behavior (temperature, etc.).
    /// - `transport_options`: Configuration for network transport (timeout, proxy, etc.).
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
pub use openai::{OpenAI, OpenAIClient, OpenAIModel};
pub use openrouter::{OpenRouter, OpenRouterClient, OpenRouterModel};
pub use perplexity::{Perplexity, PerplexityClient, PerplexityModel};
pub use together::{Together, TogetherClient, TogetherModel};
pub use xai::{XAIClient, XAIModel, XAI};
