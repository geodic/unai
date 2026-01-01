//! Core client trait and error types.

use async_trait::async_trait;
use futures::Stream;
use thiserror::Error;

use crate::model::{Response, StreamChunk, Message};
use crate::options::{ModelOptions, TransportOptions};
use rmcp::model::Tool;

/// Errors that can occur during client operations.
#[derive(Error, Debug)]
pub enum ClientError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("JSON parse error: {0}")]
    Parse(#[from] serde_json::Error),

    #[error("Provider error: {0}")]
    ProviderError(String),

    #[error("Stream cancelled")]
    StreamCancelled,

    #[error("Configuration error: {0}")]
    Config(String),
}

/// Main client trait for LLM providers.
#[async_trait]
pub trait Client: Send + Sync {
    /// Provider-specific model options type.
    type ModelProvider: Send + Sync;

    /// Send a request to the LLM provider.
    async fn request(&self, messages: Vec<Message>, tools: Vec<Tool>) -> Result<Response, ClientError>;

    /// Get reference to the model options.
    fn model_options(&self) -> &ModelOptions<Self::ModelProvider>;

    /// Get reference to the transport options.
    fn transport_options(&self) -> &TransportOptions;
}

/// Extension trait for streaming support.
#[async_trait]
pub trait StreamingClient: Client {
    /// Send a streaming request to the LLM provider.
    async fn request_stream(
        &self,
        messages: Vec<Message>,
        tools: Vec<Tool>,
    ) -> Result<std::pin::Pin<Box<dyn Stream<Item = Result<StreamChunk, ClientError>> + Send>>, ClientError>;
}
