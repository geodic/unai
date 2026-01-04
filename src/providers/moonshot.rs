//! Moonshot API client implementation.

use crate::api::openai::{OpenAiCompatibleClient, OpenAiCompatibleModel};
use crate::options::{ModelOptions, TransportOptions};
use crate::providers::Provider;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MoonshotModel;

impl OpenAiCompatibleModel for MoonshotModel {}

pub type MoonshotClient = OpenAiCompatibleClient<MoonshotModel>;

pub struct Moonshot;

impl Provider for Moonshot {
    type Client = MoonshotClient;

    fn create(api_key: String) -> Self::Client {
        MoonshotClient::new(
            api_key,
            "https://api.moonshot.cn".to_string(),
            ModelOptions::default(),
            TransportOptions::default(),
        )
    }

    fn create_with_options(
        api_key: String,
        model_options: ModelOptions<MoonshotModel>,
        transport_options: TransportOptions,
    ) -> Self::Client {
        MoonshotClient::new(
            api_key,
            "https://api.moonshot.cn".to_string(),
            model_options,
            transport_options,
        )
    }
}
