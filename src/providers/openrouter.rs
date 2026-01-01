//! OpenRouter API client implementation.

use serde::{Deserialize, Serialize};
use crate::api::openai::{OpenAiCompatibleClient, OpenAiCompatibleModel};
use crate::options::{ModelOptions, TransportOptions};
use crate::providers::Provider;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OpenRouterModel;

impl OpenAiCompatibleModel for OpenRouterModel {}

pub type OpenRouterClient = OpenAiCompatibleClient<OpenRouterModel>;

pub struct OpenRouter;

impl Provider for OpenRouter {
    type Client = OpenRouterClient;

    fn create(api_key: String) -> Self::Client {
        OpenRouterClient::new(
            api_key,
            "https://openrouter.ai/api".to_string(),
            ModelOptions::default(),
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
