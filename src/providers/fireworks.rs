//! Fireworks API client implementation.

use crate::api::openai::{OpenAiCompatibleClient, OpenAiCompatibleModel};
use crate::options::{ModelOptions, TransportOptions};
use crate::providers::Provider;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FireworksModel;

impl OpenAiCompatibleModel for FireworksModel {}

pub type FireworksClient = OpenAiCompatibleClient<FireworksModel>;

pub struct Fireworks;

impl Provider for Fireworks {
    type Client = FireworksClient;

    fn create(api_key: String) -> Self::Client {
        FireworksClient::new(
            api_key,
            "https://api.fireworks.ai/inference".to_string(),
            ModelOptions::default(),
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
