//! HTTP client utilities for making requests to LLM APIs.

use reqwest::{Client, RequestBuilder};

use crate::client::ClientError;
use crate::options::TransportOptions;

/// Build a configured HTTP client from transport options.
pub fn build_http_client(transport_options: &TransportOptions) -> Result<Client, reqwest::Error> {
    let mut builder = Client::builder();

    match transport_options {
        TransportOptions::Http { timeout, proxy, .. } => {
            if let Some(t) = timeout {
                builder = builder.timeout(*t);
            }
            if let Some(proxy_url) = proxy {
                if let Ok(p) = reqwest::Proxy::all(proxy_url) {
                    builder = builder.proxy(p);
                }
            }
        }
    }

    builder.build()
}

/// Add extra headers to a request if specified in transport options.
pub fn add_extra_headers(
    mut request: RequestBuilder,
    transport_options: &TransportOptions,
) -> RequestBuilder {
    match transport_options {
        TransportOptions::Http { headers, .. } => {
            if let Some(h) = headers {
                for (key, value) in h {
                    request = request.header(key, value);
                }
            }
        }
    }
    request
}

/// Extension trait for RequestBuilder that logs request body.
pub trait RequestBuilderExt {
    /// Set JSON request body and log it. Returns the RequestBuilder for chaining.
    fn json_logged<T: serde::Serialize + ?Sized>(self, json: &T) -> Self;
}

impl RequestBuilderExt for RequestBuilder {
    fn json_logged<T: serde::Serialize + ?Sized>(self, json: &T) -> Self {
        if let Ok(req_body) = serde_json::to_string_pretty(json) {
            tracing::debug!("API request body ({} bytes):\n{}", req_body.len(), req_body);
        }

        self.json(json)
    }
}

/// Extension trait for Response that logs response body.
#[async_trait::async_trait]
pub trait ResponseExt {
    /// Get response text and log it. Consumes the response.
    async fn text_logged(self) -> Result<String, reqwest::Error>;

    /// Parse response as JSON and log it. Consumes the response.
    async fn json_logged<T: serde::de::DeserializeOwned>(self) -> Result<T, ClientError>;
}

#[async_trait::async_trait]
impl ResponseExt for reqwest::Response {
    async fn text_logged(self) -> Result<String, reqwest::Error> {
        let text = self.text().await?;
        tracing::debug!("API response ({} bytes):\n{}", text.len(), text);
        Ok(text)
    }

    async fn json_logged<T: serde::de::DeserializeOwned>(self) -> Result<T, ClientError> {
        let bytes = self.bytes().await?;

        if let Ok(text) = std::str::from_utf8(&bytes) {
            tracing::debug!("API response ({} bytes):\n{}", text.len(), text);
        }

        serde_json::from_slice(&bytes).map_err(ClientError::from)
    }
}
