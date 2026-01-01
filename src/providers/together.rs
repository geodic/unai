//! Together AI API client implementation.

use serde::{Deserialize, Serialize};
use crate::api::openai::{OpenAiCompatibleClient, OpenAiCompatibleModel};
use crate::options::{ModelOptions, TransportOptions};
use crate::providers::Provider;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TogetherModel;

impl OpenAiCompatibleModel for TogetherModel {}

pub type TogetherClient = OpenAiCompatibleClient<TogetherModel>;

pub struct Together;

impl Provider for Together {
    type Client = TogetherClient;

    fn create(api_key: String) -> Self::Client {
        TogetherClient::new(
            api_key,
            "https://api.together.xyz".to_string(),
            ModelOptions::default(),
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
