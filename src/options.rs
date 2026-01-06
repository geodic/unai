//! Generic options structures for model and transport configuration.

use serde::{Deserialize, Serialize};
use serde_with::skip_serializing_none;
use std::collections::HashMap;
use std::time::Duration;

/// Generic model options containing common model behavior parameters
/// and provider-specific model configuration.
///
/// This struct allows configuring both universal parameters (like temperature)
/// and provider-specific parameters (via the generic `T`).
#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelOptions<T> {
    /// Model identifier (e.g., "gpt-5", "claude-4.5-opus").
    pub model: String,

    /// System instructions passed to the model.
    /// These are typically prepended to the conversation or sent as a specific system message.
    pub system: Option<String>,

    /// Enable reasoning/thinking mode (for models that support it, e.g., o1, Claude 4.5).
    pub reasoning: Option<bool>,

    /// Temperature for sampling (0.0 - 2.0).
    /// Controls randomness: lower values are more deterministic, higher values are more creative.
    pub temperature: Option<f32>,

    /// Top-p (nucleus) sampling parameter.
    /// Alternative to temperature, controls the cumulative probability of token selection.
    pub top_p: Option<f32>,

    /// Maximum tokens to generate.
    /// Limits the length of the response.
    pub max_tokens: Option<u32>,

    /// Provider-specific model options.
    /// Contains fields unique to the specific provider (e.g., `top_k` for Anthropic/Gemini).
    pub provider: T,
}

impl<T: Default> ModelOptions<T> {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            system: None,
            reasoning: None,
            temperature: None,
            top_p: None,
            max_tokens: None,
            provider: T::default(),
        }
    }
}

/// Transport configuration options.
///
/// Controls how requests are sent over the network.
#[derive(Debug, Clone)]
pub enum TransportOptions {
    /// HTTP transport configuration
    Http {
        /// Request timeout. If None, default client timeout is used.
        timeout: Option<Duration>,
        /// HTTP proxy URL.
        proxy: Option<String>,
        /// Additional HTTP headers to send with every request.
        headers: Option<HashMap<String, String>>,
    },
}

impl Default for TransportOptions {
    fn default() -> Self {
        TransportOptions::Http {
            timeout: None,
            proxy: None,
            headers: None,
        }
    }
}

impl TransportOptions {
    /// Create new default HTTP transport options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the timeout.
    pub fn with_timeout(mut self, duration: Duration) -> Self {
        match &mut self {
            TransportOptions::Http { timeout, .. } => *timeout = Some(duration),
        }
        self
    }

    /// Set the proxy.
    pub fn with_proxy(mut self, proxy_url: String) -> Self {
        match &mut self {
            TransportOptions::Http { proxy, .. } => *proxy = Some(proxy_url),
        }
        self
    }

    /// Add a header.
    pub fn with_header(mut self, key: String, value: String) -> Self {
        match &mut self {
            TransportOptions::Http { headers, .. } => {
                headers.get_or_insert_with(HashMap::new).insert(key, value);
            }
        }
        self
    }
}
