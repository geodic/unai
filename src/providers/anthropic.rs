//! Anthropic API client implementation.

pub use crate::api::anthropic::{AnthropicClient, AnthropicModel};
use crate::options::{ModelOptions, TransportOptions};
use crate::providers::Provider;

pub struct Anthropic;

impl Provider for Anthropic {
    type Client = AnthropicClient;

    fn create(api_key: String, model: String) -> Self::Client {
        Self::create_with_options(
            api_key,
            ModelOptions::new(model),
            TransportOptions::default(),
        )
    }

    fn create_with_options(
        api_key: String,
        model_options: ModelOptions<AnthropicModel>,
        transport_options: TransportOptions,
    ) -> Self::Client {
        AnthropicClient::new(
            api_key,
            "https://api.anthropic.com/v1".to_string(),
            model_options,
            transport_options,
        )
    }
}
