//! DeepSeek API client implementation.

use crate::api::openai::{OpenAiCompatibleClient, OpenAiCompatibleModel};
use crate::options::{ModelOptions, TransportOptions};
use crate::providers::Provider;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DeepSeekModel;

impl OpenAiCompatibleModel for DeepSeekModel {}

pub type DeepSeekClient = OpenAiCompatibleClient<DeepSeekModel>;

pub struct DeepSeek;

impl Provider for DeepSeek {
    type Client = DeepSeekClient;

    fn create(api_key: String) -> Self::Client {
        DeepSeekClient::new(
            api_key,
            "https://api.deepseek.com".to_string(),
            ModelOptions::default(),
            TransportOptions::default(),
        )
    }

    fn create_with_options(
        api_key: String,
        model_options: ModelOptions<DeepSeekModel>,
        transport_options: TransportOptions,
    ) -> Self::Client {
        DeepSeekClient::new(
            api_key,
            "https://api.deepseek.com".to_string(),
            model_options,
            transport_options,
        )
    }
}
