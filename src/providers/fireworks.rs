//! Fireworks API client implementation.

use crate::api::openai::{OpenAIClient, OpenAICompatibleModel};
use crate::options::{ModelOptions, TransportOptions};
use crate::providers::Provider;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FireworksModel;

impl OpenAICompatibleModel for FireworksModel {}

pub type FireworksClient = OpenAIClient<FireworksModel>;

pub struct Fireworks;

impl Provider for Fireworks {
    type Client = FireworksClient;

    fn create(api_key: String, model: String) -> Self::Client {
        Self::create_with_options(
            api_key,
            ModelOptions::new(model),
            TransportOptions::default(),
        )
    }

    fn create_with_options(
        api_key: String,
        model_options: ModelOptions<FireworksModel>,
        transport_options: TransportOptions,
    ) -> Self::Client {
        FireworksClient::new(
            api_key,
            "https://api.fireworks.ai/inference".to_string(),
            model_options,
            transport_options,
        )
    }
}
