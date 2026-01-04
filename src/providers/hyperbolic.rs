//! Hyperbolic API client implementation.

use crate::api::openai::{OpenAiCompatibleClient, OpenAiCompatibleModel};
use crate::options::{ModelOptions, TransportOptions};
use crate::providers::Provider;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HyperbolicModel;

impl OpenAiCompatibleModel for HyperbolicModel {}

pub type HyperbolicClient = OpenAiCompatibleClient<HyperbolicModel>;

pub struct Hyperbolic;

impl Provider for Hyperbolic {
    type Client = HyperbolicClient;

    fn create(api_key: String) -> Self::Client {
        HyperbolicClient::new(
            api_key,
            "https://api.hyperbolic.xyz".to_string(),
            ModelOptions::default(),
            TransportOptions::default(),
        )
    }

    fn create_with_options(
        api_key: String,
        model_options: ModelOptions<HyperbolicModel>,
        transport_options: TransportOptions,
    ) -> Self::Client {
        HyperbolicClient::new(
            api_key,
            "https://api.hyperbolic.xyz".to_string(),
            model_options,
            transport_options,
        )
    }
}
