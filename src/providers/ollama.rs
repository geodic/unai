//! Ollama API client implementation.

use serde::{Deserialize, Serialize};
use crate::api::openai::{OpenAiCompatibleClient, OpenAiCompatibleModel};
use crate::options::{ModelOptions, TransportOptions};
use crate::providers::Provider;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OllamaModel;

impl OpenAiCompatibleModel for OllamaModel {}

pub type OllamaClient = OpenAiCompatibleClient<OllamaModel>;

pub struct Ollama;

impl Provider for Ollama {
    type Client = OllamaClient;

    fn create(base_url: String) -> Self::Client {
        OllamaClient::new(
            "ollama".to_string(),
            base_url,
            ModelOptions::default(),
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
