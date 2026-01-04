//! Generic options structures for model and transport configuration.

use serde::{Deserialize, Serialize};
use serde_with::skip_serializing_none;
use std::collections::HashMap;
use std::time::Duration;

/// Generic model options containing common model behavior parameters
/// and provider-specific model configuration.
#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelOptions<T> {
    /// Model identifier (e.g., "gpt-4o", "claude-3-opus")
    pub model: Option<String>,

    // System instructions passed to the model
    pub system: Option<String>,

    /// Enable reasoning/thinking mode (for models that support it)
    pub reasoning: Option<bool>,

    /// Temperature for sampling (0.0 - 2.0)
    pub temperature: Option<f32>,

    /// Top-p (nucleus) sampling parameter
    pub top_p: Option<f32>,

    /// Maximum tokens to generate
    pub max_tokens: Option<u32>,

    /// Provider-specific model options
    pub provider: T,
}

/// Transport configuration options.
#[derive(Debug, Clone)]
pub enum TransportOptions {
    /// HTTP transport configuration
    Http {
        /// Request timeout
        timeout: Option<Duration>,
        /// HTTP proxy URL
        proxy: Option<String>,
        /// Additional HTTP headers
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
