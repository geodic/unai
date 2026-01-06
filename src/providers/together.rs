//! Together AI API client implementation.

use crate::api::openai::{OpenAIClient, OpenAICompatibleModel};
use crate::options::{ModelOptions, TransportOptions};
use crate::providers::Provider;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TogetherModel;

impl OpenAICompatibleModel for TogetherModel {}

pub type TogetherClient = OpenAIClient<TogetherModel>;

pub struct Together;

impl Provider for Together {
    type Client = TogetherClient;

    fn create(api_key: String, model: String) -> Self::Client {
        Self::create_with_options(
            api_key,
            ModelOptions::new(model),
            TransportOptions::default(),
        )
    }

    fn create_with_options(
        api_key: String,
        model_options: ModelOptions<TogetherModel>,
        transport_options: TransportOptions,
    ) -> Self::Client {
        TogetherClient::new(
            api_key,
            "https://api.together.xyz".to_string(),
            model_options,
            transport_options,
        )
    }
}
