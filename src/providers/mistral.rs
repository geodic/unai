//! Mistral API client implementation.

use crate::api::openai::{OpenAIClient, OpenAICompatibleModel};
use crate::options::{ModelOptions, TransportOptions};
use crate::providers::Provider;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MistralModel;

impl OpenAICompatibleModel for MistralModel {}

pub type MistralClient = OpenAIClient<MistralModel>;

pub struct Mistral;

impl Provider for Mistral {
    type Client = MistralClient;

    fn create(api_key: String, model: String) -> Self::Client {
        Self::create_with_options(
            api_key,
            ModelOptions::new(model),
            TransportOptions::default(),
        )
    }

    fn create_with_options(
        api_key: String,
        model_options: ModelOptions<MistralModel>,
        transport_options: TransportOptions,
    ) -> Self::Client {
        MistralClient::new(
            api_key,
            "https://api.mistral.ai".to_string(),
            model_options,
            transport_options,
        )
    }
}
