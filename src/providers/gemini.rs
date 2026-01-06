//! Gemini provider implementation.

use crate::options::{ModelOptions, TransportOptions};
use crate::providers::Provider;

pub use crate::api::gemini::{GeminiClient, GeminiModel};

pub struct Gemini;

impl Provider for Gemini {
    type Client = GeminiClient;

    fn create(api_key: String, model: String) -> Self::Client {
        Self::create_with_options(
            api_key,
            ModelOptions::new(model),
            TransportOptions::default(),
        )
    }

    fn create_with_options(
        api_key: String,
        model_options: ModelOptions<GeminiModel>,
        transport_options: TransportOptions,
    ) -> Self::Client {
        GeminiClient::new(
            api_key,
            "https://generativelanguage.googleapis.com/v1beta".to_string(),
            model_options,
            transport_options,
        )
    }
}
