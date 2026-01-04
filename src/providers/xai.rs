//! xAI API client implementation.

use crate::api::openai::{OpenAiCompatibleClient, OpenAiCompatibleModel};
use crate::options::{ModelOptions, TransportOptions};
use crate::providers::Provider;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct XAIModel;

impl OpenAiCompatibleModel for XAIModel {}

pub type XAIClient = OpenAiCompatibleClient<XAIModel>;

pub struct XAI;

impl Provider for XAI {
    type Client = XAIClient;

    fn create(api_key: String) -> Self::Client {
        XAIClient::new(
            api_key,
            "https://api.x.ai".to_string(),
            ModelOptions::default(),
            TransportOptions::default(),
        )
    }

    fn create_with_options(
        api_key: String,
        model_options: ModelOptions<XAIModel>,
        transport_options: TransportOptions,
    ) -> Self::Client {
        XAIClient::new(
            api_key,
            "https://api.x.ai".to_string(),
            model_options,
            transport_options,
        )
    }
}
