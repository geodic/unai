//! OpenAI API client implementation.

use crate::api::openai::{OpenAIClient as GenericOpenAIClient, OpenAICompatibleModel};
use crate::options::{ModelOptions, TransportOptions};
use crate::providers::Provider;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OpenAIModel;

impl OpenAICompatibleModel for OpenAIModel {}

pub type OpenAIClient = GenericOpenAIClient<OpenAIModel>;

pub struct OpenAI;

impl Provider for OpenAI {
    type Client = OpenAIClient;

    fn create(api_key: String, model: String) -> Self::Client {
        Self::create_with_options(
            api_key,
            ModelOptions::new(model),
            TransportOptions::default(),
        )
    }

    fn create_with_options(
        api_key: String,
        model_options: ModelOptions<OpenAIModel>,
        transport_options: TransportOptions,
    ) -> Self::Client {
        OpenAIClient::new(
            api_key,
            "https://api.openai.com".to_string(),
            model_options,
            transport_options,
        )
    }
}
