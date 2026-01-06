//! Ollama API client implementation.

use crate::api::openai::{OpenAIClient, OpenAICompatibleModel};
use crate::options::{ModelOptions, TransportOptions};
use crate::providers::Provider;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OllamaModel;

impl OpenAICompatibleModel for OllamaModel {}

pub type OllamaClient = OpenAIClient<OllamaModel>;

pub struct Ollama;

impl Provider for Ollama {
    type Client = OllamaClient;

    fn create(base_url: String, model: String) -> Self::Client {
        Self::create_with_options(
            base_url,
            ModelOptions::new(model),
            TransportOptions::default(),
        )
    }

    fn create_with_options(
        base_url: String,
        model_options: ModelOptions<OllamaModel>,
        transport_options: TransportOptions,
    ) -> Self::Client {
        OllamaClient::new(
            "ollama".to_string(),
            base_url,
            model_options,
            transport_options,
        )
    }
}
