//! Anthropic API client implementation.

use async_trait::async_trait;
use futures::Stream;
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};
use serde::{Deserialize, Serialize};
use serde_with::skip_serializing_none;

use crate::client::{Client, ClientError, StreamingClient};
use crate::http::{add_extra_headers, build_http_client, RequestBuilderExt, ResponseExt};
use crate::model::{FinishReason, Message, Response, Role, Usage, Part};
use crate::options::{ModelOptions, TransportOptions};

const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Anthropic model options.
#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AnthropicModel {
    pub top_k: Option<u32>,
    pub metadata: Option<serde_json::Value>,
}

/// Anthropic client.
#[derive(Debug, Clone)]
pub struct AnthropicClient {
    api_key: String,
    base_url: String,
    model_options: ModelOptions<AnthropicModel>,
    transport_options: TransportOptions,
}

impl AnthropicClient {
    pub fn new(
        api_key: String,
        base_url: String,
        model_options: ModelOptions<AnthropicModel>,
        transport_options: TransportOptions,
    ) -> Self {
        Self {
            api_key,
            base_url,
            model_options,
            transport_options,
        }
    }

    fn handle_error_response(status: reqwest::StatusCode, body: &str) -> ClientError {
        if let Ok(error_resp) = serde_json::from_str::<AnthropicErrorResponse>(body) {
            ClientError::ProviderError(format!(
                "Anthropic error ({}): {}",
                error_resp.error.error_type, error_resp.error.message
            ))
        } else {
            ClientError::ProviderError(format!("HTTP {}: {}", status, body))
        }
    }
}

#[async_trait]
impl Client for AnthropicClient {
    type ModelProvider = AnthropicModel;

    async fn request(
        &self,
        messages: Vec<Message>,
        tools: Vec<rmcp::model::Tool>,
    ) -> Result<Response, ClientError> {
        let url = format!("{}/messages", self.base_url);
        
        let model = self.model_options.model.clone()
            .ok_or_else(|| ClientError::Config("Model must be specified".to_string()))?;

        let request_body = AnthropicRequest::new(messages, &self.model_options, model, tools);

        let http_client = build_http_client(&self.transport_options)?;

        let mut headers = HeaderMap::new();
        headers.insert("x-api-key", HeaderValue::from_str(&self.api_key).map_err(|_| ClientError::Config("Invalid API key".to_string()))?);
        headers.insert("anthropic-version", HeaderValue::from_static(ANTHROPIC_VERSION));
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        let mut req = http_client
            .post(&url)
            .headers(headers);

        req = add_extra_headers(req, &self.transport_options);

        let response = req.json_logged(&request_body).send().await?;
        let status = response.status();

        if !status.is_success() {
            let body = response.text_logged().await.unwrap_or_default();
            return Err(Self::handle_error_response(status, &body));
        }

        let anthropic_response: AnthropicResponse = response.json_logged().await?;
        Ok(anthropic_response.into())
    }

    fn model_options(&self) -> &ModelOptions<Self::ModelProvider> {
        &self.model_options
    }

    fn transport_options(&self) -> &TransportOptions {
        &self.transport_options
    }
}

#[async_trait]
impl StreamingClient for AnthropicClient {
    async fn request_stream(
        &self,
        _messages: Vec<Message>,
        _tools: Vec<rmcp::model::Tool>,
    ) -> Result<
        std::pin::Pin<Box<dyn Stream<Item = Result<crate::model::StreamChunk, ClientError>> + Send>>,
        ClientError,
    > {
        // Placeholder for streaming implementation
        Err(ClientError::ProviderError("Streaming not yet implemented for Anthropic".to_string()))
    }
}

// --- Request Types ---

#[skip_serializing_none]
#[derive(Debug, Serialize)]
struct AnthropicRequest {
    model: String,
    messages: Vec<AnthropicMessage>,
    max_tokens: u32,
    system: Option<String>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<AnthropicTool>,
    #[serde(flatten)]
    provider_options: AnthropicModel,
}

#[derive(Debug, Serialize)]
struct AnthropicTool {
    name: String,
    description: Option<String>,
    input_schema: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct AnthropicMessage {
    role: String,
    content: String,
}

impl AnthropicRequest {
    fn new(
        messages_in: Vec<Message>,
        model_options: &ModelOptions<AnthropicModel>,
        model: String,
        tool_defs: Vec<rmcp::model::Tool>,
    ) -> Self {
        let mut messages = Vec::new();
        let mut system_prompt = model_options.instructions.clone();

        for msg in &messages_in {
            match msg {
                Message::User(parts) => {
                    let content = parts.iter().filter_map(|p| match p {
                        Part::Text(t) => Some(t.clone()),
                        _ => None,
                    }).collect::<Vec<_>>().join("\n");
                    
                    if !content.is_empty() {
                        messages.push(AnthropicMessage {
                            role: "user".to_string(),
                            content,
                        });
                    }
                }
                Message::Assistant(parts) => {
                    let content = parts.iter().filter_map(|p| match p {
                        Part::Text(t) => Some(t.clone()),
                        _ => None,
                    }).collect::<Vec<_>>().join("\n");

                    if !content.is_empty() {
                        messages.push(AnthropicMessage {
                            role: "assistant".to_string(),
                            content,
                        });
                    }
                }
                Message::System(parts) => {
                    let content = parts.iter().filter_map(|p| match p {
                        Part::Text(t) => Some(t.clone()),
                        _ => None,
                    }).collect::<Vec<_>>().join("\n");
                    
                    if !content.is_empty() {
                        if let Some(sys) = &mut system_prompt {
                            sys.push_str("\n");
                            sys.push_str(&content);
                        } else {
                            system_prompt = Some(content);
                        }
                    }
                }
            }
        }

        let tools = tool_defs.into_iter().map(|t| AnthropicTool {
            name: t.name.into_owned(),
            description: t.description.map(|d| d.into_owned()),
            input_schema: serde_json::Value::Object((*t.input_schema).clone()),
        }).collect();

        AnthropicRequest {
            model,
            messages,
            max_tokens: model_options.max_tokens.unwrap_or(1024),
            system: system_prompt,
            temperature: model_options.temperature,
            top_p: model_options.top_p,
            stream: None,
            tools,
            provider_options: model_options.provider.clone(),
        }
    }
}

// --- Response Types ---

#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    id: String,
    #[serde(rename = "type")]
    msg_type: String,
    role: String,
    content: Vec<AnthropicContent>,
    model: String,
    stop_reason: Option<String>,
    stop_sequence: Option<String>,
    usage: AnthropicUsage,
}

#[derive(Debug, Deserialize)]
struct AnthropicContent {
    #[serde(rename = "type")]
    content_type: String,
    text: String,
}

#[derive(Debug, Deserialize)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct AnthropicErrorResponse {
    error: AnthropicError,
}

#[derive(Debug, Deserialize)]
struct AnthropicError {
    #[serde(rename = "type")]
    error_type: String,
    message: String,
}

impl From<AnthropicResponse> for Response {
    fn from(resp: AnthropicResponse) -> Self {
        let content = resp.content.into_iter()
            .map(|c| c.text)
            .collect::<Vec<_>>()
            .join("");

        let finish_reason = match resp.stop_reason.as_deref() {
            Some("end_turn") => FinishReason::Stop,
            Some("max_tokens") => FinishReason::OutputTokens,
            Some("stop_sequence") => FinishReason::Stop,
            _ => FinishReason::Stop,
        };

        Response {
            data: vec![Message::Assistant(vec![Part::Text(content)])],
            usage: Some(Usage {
                prompt_tokens: Some(resp.usage.input_tokens),
                completion_tokens: Some(resp.usage.output_tokens),
            }),
            finish: finish_reason,
        }
    }
}
