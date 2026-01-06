//! OpenRouter API client implementation.

use crate::api::openai::{OpenAIClient, OpenAICompatibleModel};
use crate::options::{ModelOptions, TransportOptions};
use crate::providers::Provider;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OpenRouterModel;

impl OpenAICompatibleModel for OpenRouterModel {}

pub type OpenRouterClient = OpenAIClient<OpenRouterModel>;

pub struct OpenRouter;

impl Provider for OpenRouter {
    type Client = OpenRouterClient;

    fn create(api_key: String, model: String) -> Self::Client {
        Self::create_with_options(
            api_key,
            ModelOptions::new(model),
            TransportOptions::default(),
        )
    }

    fn create_with_options(
        api_key: String,
        model_options: ModelOptions<OpenRouterModel>,
        transport_options: TransportOptions,
    ) -> Self::Client {
        OpenRouterClient::new(
            api_key,
            "https://openrouter.ai/api".to_string(),
            model_options,
            transport_options,
        )
    }
}
