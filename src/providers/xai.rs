//! xAI API client implementation.

use crate::api::openai::{OpenAIClient, OpenAICompatibleModel};
use crate::options::{ModelOptions, TransportOptions};
use crate::providers::Provider;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct XAIModel;

impl OpenAICompatibleModel for XAIModel {}

pub type XAIClient = OpenAIClient<XAIModel>;

pub struct XAI;

impl Provider for XAI {
    type Client = XAIClient;

    fn create(api_key: String, model: String) -> Self::Client {
        Self::create_with_options(
            api_key,
            ModelOptions::new(model),
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
