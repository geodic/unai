//! OpenAI API client implementation.

use crate::api::openai::{OpenAiCompatibleClient, OpenAiCompatibleModel};
use crate::options::{ModelOptions, TransportOptions};
use crate::providers::Provider;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OpenAiModel;

impl OpenAiCompatibleModel for OpenAiModel {}

pub type OpenAiClient = OpenAiCompatibleClient<OpenAiModel>;

pub struct OpenAi;

impl Provider for OpenAi {
    type Client = OpenAiClient;

    fn create(api_key: String) -> Self::Client {
        OpenAiClient::new(
            api_key,
            "https://api.openai.com".to_string(),
            ModelOptions::default(),
            TransportOptions::default(),
        )
    }

    fn create_with_options(
        api_key: String,
        model_options: ModelOptions<OpenAiModel>,
        transport_options: TransportOptions,
    ) -> Self::Client {
        OpenAiClient::new(
            api_key,
            "https://api.openai.com".to_string(),
            model_options,
            transport_options,
        )
    }
}
