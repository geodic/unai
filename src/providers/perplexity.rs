//! Perplexity API client implementation.

use crate::api::openai::{OpenAIClient, OpenAICompatibleModel};
use crate::options::{ModelOptions, TransportOptions};
use crate::providers::Provider;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PerplexityModel;

impl OpenAICompatibleModel for PerplexityModel {}

pub type PerplexityClient = OpenAIClient<PerplexityModel>;

pub struct Perplexity;

impl Provider for Perplexity {
    type Client = PerplexityClient;

    fn create(api_key: String, model: String) -> Self::Client {
        Self::create_with_options(
            api_key,
            ModelOptions::new(model),
            TransportOptions::default(),
        )
    }

    fn create_with_options(
        api_key: String,
        model_options: ModelOptions<PerplexityModel>,
        transport_options: TransportOptions,
    ) -> Self::Client {
        PerplexityClient::new(
            api_key,
            "https://api.perplexity.ai".to_string(),
            model_options,
            transport_options,
        )
    }
}
