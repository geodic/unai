//! Groq API client implementation.

use crate::api::openai::{OpenAiCompatibleClient, OpenAiCompatibleModel};
use crate::options::{ModelOptions, TransportOptions};
use crate::providers::Provider;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GroqModel;

impl OpenAiCompatibleModel for GroqModel {}

pub type GroqClient = OpenAiCompatibleClient<GroqModel>;

pub struct Groq;

impl Provider for Groq {
    type Client = GroqClient;

    fn create(api_key: String) -> Self::Client {
        GroqClient::new(
            api_key,
            "https://api.groq.com/openai/v1".to_string(),
            ModelOptions::default(),
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
