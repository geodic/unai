//! Groq API client implementation.

use crate::api::openai::{OpenAIClient, OpenAICompatibleModel};
use crate::options::{ModelOptions, TransportOptions};
use crate::providers::Provider;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GroqModel;

impl OpenAICompatibleModel for GroqModel {}

pub type GroqClient = OpenAIClient<GroqModel>;

pub struct Groq;

impl Provider for Groq {
    type Client = GroqClient;

    fn create(api_key: String, model: String) -> Self::Client {
        Self::create_with_options(
            api_key,
            ModelOptions::new(model),
            TransportOptions::default(),
        )
    }

    fn create_with_options(
        api_key: String,
        model_options: ModelOptions<GroqModel>,
        transport_options: TransportOptions,
    ) -> Self::Client {
        GroqClient::new(
            api_key,
            "https://api.groq.com/openai/v1".to_string(),
            model_options,
            transport_options,
        )
    }
}
