//! OpenAI Chat Completions API client implementation.

use async_trait::async_trait;
use futures::Stream;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::client::{Client, ClientError, StreamingClient};
use crate::http::{add_extra_headers, build_http_client, RequestBuilderExt, ResponseExt};
use crate::model::{FinishReason, Message, Response, Role, Usage, Part};
use crate::options::{ModelOptions, TransportOptions};
use crate::sse::SSEResponseExt;

/// Trait for models compatible with OpenAI's Chat Completions API.
pub trait OpenAiCompatibleModel:
    Send + Sync + Default + Serialize + for<'de> Deserialize<'de> + Clone
{
}

/// Generic client for OpenAI-compatible Chat Completions APIs.
#[derive(Debug, Clone)]
pub struct OpenAiCompatibleClient<M> {
    api_key: String,
    base_url: String,
    model_options: ModelOptions<M>,
    transport_options: TransportOptions,
}

impl<M: OpenAiCompatibleModel> OpenAiCompatibleClient<M> {
    /// Create a new client.
    pub fn new(
        api_key: String,
        base_url: String,
        model_options: ModelOptions<M>,
        transport_options: TransportOptions,
    ) -> Self {
        Self {
            api_key,
            base_url,
            model_options,
            transport_options,
        }
    }

    /// Process streaming response.
    fn process_stream(
        response: reqwest::Response,
    ) -> impl Stream<Item = Result<crate::model::StreamChunk, ClientError>> + Send {
        use crate::model::StreamChunk;
        use futures::StreamExt;

        let sse_stream = response.sse().map(|result| {
            result.and_then(|line| {
                if line == "[DONE]" {
                    return Ok(ChatStreamEvent::Done);
                }
                serde_json::from_str::<ChatStreamEvent>(&line).map_err(ClientError::Parse)
            })
        });

        use futures::stream;

        sse_stream.flat_map(|result| {
            match result {
                Ok(event) => match event {
                    ChatStreamEvent::Chunk(chunk) => {
                        let mut chunks = Vec::new();

                        if let Some(choice) = chunk.choices.first() {
                            if let Some(delta) = &choice.delta {
                                // Text content delta
                                if let Some(content) = &delta.content {
                                    chunks.push(Ok(StreamChunk::Data(Message::Assistant(vec![
                                        Part::Text(content.clone())
                                    ]))));
                                }

                                // Tool calls
                                if let Some(tool_calls) = &delta.tool_calls {
                                    for tool_call in tool_calls {
                                        if let Some(function) = &tool_call.function {
                                            if let Some(name) = &function.name {
                                                // We have a complete function call
                                                chunks.push(Ok(StreamChunk::Data(
                                                    Message::Assistant(vec![Part::FunctionCall {
                                                        id: tool_call.id.clone(),
                                                        name: name.clone(),
                                                        arguments: Value::String(function.arguments.clone().unwrap_or_default()),
                                                        signature: None,
                                                    }]),
                                                )));
                                            }
                                        }
                                    }
                                }
                            }

                            // Finish reason
                            if let Some(finish_reason) = &choice.finish_reason {
                                let reason = match finish_reason.as_str() {
                                    "stop" => FinishReason::Stop,
                                    "length" => FinishReason::OutputTokens,
                                    "content_filter" => FinishReason::ContentFilter,
                                    "tool_calls" => FinishReason::ToolCalls,
                                    _ => FinishReason::Stop,
                                };
                                chunks.push(Ok(StreamChunk::Finish(reason)));
                            }
                        }

                        // Usage info
                        if let Some(usage) = chunk.usage {
                            chunks.push(Ok(StreamChunk::Usage(Usage {
                                prompt_tokens: Some(usage.prompt_tokens),
                                completion_tokens: Some(usage.completion_tokens),
                            })));
                        }

                        stream::iter(chunks)
                    }
                    ChatStreamEvent::Done => stream::iter(vec![]),
                },
                Err(e) => stream::iter(vec![Err(e)]),
            }
        })
    }

    /// Handle error responses.
    fn handle_error_response(status: reqwest::StatusCode, body: &str) -> ClientError {
        if let Ok(error_resp) = serde_json::from_str::<ChatErrorResponse>(body) {
            ClientError::ProviderError(format!(
                "API error ({}): {}",
                error_resp.error.error_type, error_resp.error.message
            ))
        } else {
            ClientError::ProviderError(format!("HTTP {}: {}", status, body))
        }
    }
}

#[async_trait]
impl<M: OpenAiCompatibleModel> Client for OpenAiCompatibleClient<M> {
    type ModelProvider = M;

    async fn request(
        &self,
        messages: Vec<Message>,
        tools: Vec<rmcp::model::Tool>,
    ) -> Result<Response, ClientError> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        
        // Model is required
        let model = self.model_options.model.clone()
            .ok_or_else(|| ClientError::Config("Model must be specified".to_string()))?;

        let request_body = ChatRequest::new(messages, &self.model_options, model, tools);

        let http_client = build_http_client(&self.transport_options)?;

        let mut req = http_client
            .post(&url)
            .header(AUTHORIZATION, format!("Bearer {}", self.api_key))
            .header(CONTENT_TYPE, "application/json");

        req = add_extra_headers(req, &self.transport_options);

        let response = req.json_logged(&request_body).send().await?;
        let status = response.status();

        if !status.is_success() {
            let body = response.text_logged().await.unwrap_or_default();
            return Err(Self::handle_error_response(status, &body));
        }

        let chat_response: ChatResponse = response.json_logged().await?;
        Ok(chat_response.into())
    }

    fn model_options(&self) -> &ModelOptions<Self::ModelProvider> {
        &self.model_options
    }

    fn transport_options(&self) -> &TransportOptions {
        &self.transport_options
    }
}

#[async_trait]
impl<M: OpenAiCompatibleModel + 'static> StreamingClient for OpenAiCompatibleClient<M> {
    async fn request_stream(
        &self,
        messages: Vec<Message>,
        tools: Vec<rmcp::model::Tool>,
    ) -> Result<
        std::pin::Pin<Box<dyn Stream<Item = Result<crate::model::StreamChunk, ClientError>> + Send>>,
        ClientError,
    > {
        let url = format!("{}/v1/chat/completions", self.base_url);
        
        // Model is required
        let model = self.model_options.model.clone()
            .ok_or_else(|| ClientError::Config("Model must be specified".to_string()))?;

        let mut request_body = ChatRequest::new(messages, &self.model_options, model, tools);
        request_body.stream = Some(true);
        request_body.stream_options = Some(StreamOptions {
            include_usage: true,
        });

        let http_client = build_http_client(&self.transport_options)?;

        let mut req = http_client
            .post(&url)
            .header(AUTHORIZATION, format!("Bearer {}", self.api_key))
            .header(CONTENT_TYPE, "application/json");

        req = add_extra_headers(req, &self.transport_options);

        let response = req.json_logged(&request_body).send().await?;
        let status = response.status();

        if !status.is_success() {
            let body = response.text_logged().await.unwrap_or_default();
            return Err(Self::handle_error_response(status, &body));
        }

        Ok(Box::pin(Self::process_stream(response)))
    }
}

impl<M: OpenAiCompatibleModel + Clone> ChatRequest<M> {
    fn new(messages_in: Vec<Message>, model_options: &ModelOptions<M>, model: String, tool_defs: Vec<rmcp::model::Tool>) -> Self {
        // Convert messages to Chat Completions format
        let messages: Vec<ChatMessage> = messages_in
            .iter()
            .map(|msg| msg.clone().into())
            .collect();

        // Convert tool definitions if present
        let tools = if tool_defs.is_empty() {
            None
        } else {
            Some(
                tool_defs
                    .iter()
                    .map(|def| ChatTool {
                        tool_type: "function".to_string(),
                        function: ChatFunction {
                            name: def.name.to_string(),
                            description: def.description.as_ref().map(|d| d.to_string()),
                            parameters: serde_json::Value::Object((*def.input_schema).clone()),
                        },
                    })
                    .collect(),
            )
        };

        ChatRequest {
            model,
            messages,
            temperature: model_options.temperature,
            top_p: model_options.top_p,
            max_tokens: model_options.max_tokens,
            stream: None,
            stream_options: None,
            tools,
            provider_options: model_options.provider.clone(),
        }
    }
}

impl From<Message> for ChatMessage {
    fn from(msg: Message) -> Self {
        let role = match msg {
            Message::User(_) => "user",
            Message::Assistant(_) => "assistant",
            Message::System(_) => "system",
        };

        let mut content_parts = Vec::new();
        let mut tool_calls = Vec::new();
        let mut tool_call_id = None;

        for part in msg.parts() {
            match part {
                Part::Text(text) => content_parts.push(text.clone()),
                Part::Reasoning { content, .. } => content_parts.push(content.clone()),
                Part::FunctionCall { id, name, arguments, .. } => {
                    tool_calls.push(ChatToolCall {
                        id: id.clone().unwrap_or_else(|| "call_unknown".to_string()),
                        tool_type: "function".to_string(),
                        function: ChatFunctionCall {
                            name: name.clone(),
                            arguments: match arguments {
                                Value::String(s) => s.clone(),
                                _ => arguments.to_string(),
                            },
                        },
                    });
                }
                Part::FunctionResponse { id, response, .. } => {
                    content_parts.push(response.to_string());
                    tool_call_id = id.clone();
                }
                Part::Image { .. } => {
                    // Placeholder for image support
                }
            }
        }

        let final_role = if tool_call_id.is_some() { "tool" } else { role };

        ChatMessage {
            role: final_role.to_string(),
            content: if content_parts.is_empty() { None } else { Some(content_parts.join("\n")) },
            tool_calls: if tool_calls.is_empty() { None } else { Some(tool_calls) },
            tool_call_id,
        }
    }
}

impl From<ChatMessage> for Message {
    fn from(msg: ChatMessage) -> Self {
        let mut parts = Vec::new();

        if let Some(content) = msg.content {
            parts.push(Part::Text(content));
        }

        if let Some(tool_calls) = msg.tool_calls {
            for tool_call in tool_calls {
                parts.push(Part::FunctionCall {
                    id: Some(tool_call.id),
                    name: tool_call.function.name,
                    arguments: serde_json::from_str(&tool_call.function.arguments)
                        .unwrap_or(Value::Null),
                    signature: None,
                });
            }
        }

        if msg.role == "tool" {
            // Function response
            return Message::User(vec![Part::FunctionResponse {
                id: msg.tool_call_id,
                name: "unknown".to_string(),
                response: parts.first().and_then(|p| match p {
                    Part::Text(s) => serde_json::from_str(s).ok(),
                    _ => None,
                }).unwrap_or(Value::Null),
            }]);
        }

        match msg.role.as_str() {
            "user" => Message::User(parts),
            "system" => Message::System(parts),
            _ => Message::Assistant(parts),
        }
    }
}

impl From<ChatResponse> for Response {
    fn from(chat_resp: ChatResponse) -> Self {
        // Extract finish reason before consuming choices
        let finish_reason = chat_resp
            .choices
            .first()
            .and_then(|c| c.finish_reason.as_ref())
            .map(|reason| match reason.as_str() {
                "stop" => FinishReason::Stop,
                "length" => FinishReason::OutputTokens,
                "content_filter" => FinishReason::ContentFilter,
                "tool_calls" => FinishReason::ToolCalls,
                _ => FinishReason::Stop,
            })
            .unwrap_or(FinishReason::Stop);

        let messages = chat_resp
            .choices
            .into_iter()
            .map(|choice| choice.message.into())
            .collect();

        Response {
            data: messages,
            usage: chat_resp.usage.map(|u| Usage {
                prompt_tokens: Some(u.prompt_tokens),
                completion_tokens: Some(u.completion_tokens),
            }),
            finish: finish_reason,
        }
    }
}

// --- Chat Completions API Types ---

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize)]
struct ChatRequest<M> {
    model: String,
    messages: Vec<ChatMessage>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    max_tokens: Option<u32>,
    stream: Option<bool>,
    stream_options: Option<StreamOptions>,
    tools: Option<Vec<ChatTool>>,
    #[serde(flatten)]
    provider_options: M,
}

#[derive(Debug, Clone, Serialize)]
struct StreamOptions {
    include_usage: bool,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ChatMessage {
    role: String,
    content: Option<String>,
    tool_calls: Option<Vec<ChatToolCall>>,
    tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ChatToolCall {
    id: String,
    #[serde(rename = "type")]
    tool_type: String,
    function: ChatFunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ChatFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Debug, Clone, Serialize)]
struct ChatTool {
    #[serde(rename = "type")]
    tool_type: String,
    function: ChatFunction,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize)]
struct ChatFunction {
    name: String,
    description: Option<String>,
    parameters: serde_json::Value,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
    usage: Option<ChatUsage>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize)]
struct ChatChoice {
    message: ChatMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct ChatUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

#[derive(Debug, Clone, Deserialize)]
struct ChatErrorResponse {
    error: ChatError,
}

#[derive(Debug, Clone, Deserialize)]
struct ChatError {
    #[serde(rename = "type")]
    error_type: String,
    message: String,
}

// --- Streaming Types ---

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum ChatStreamEvent {
    Chunk(ChatStreamChunk),
    Done,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize)]
struct ChatStreamChunk {
    choices: Vec<ChatStreamChoice>,
    usage: Option<ChatUsage>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize)]
struct ChatStreamChoice {
    delta: Option<ChatDelta>,
    finish_reason: Option<String>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize)]
struct ChatDelta {
    content: Option<String>,
    tool_calls: Option<Vec<ChatToolCallDelta>>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize)]
struct ChatToolCallDelta {
    id: Option<String>,
    function: Option<ChatFunctionCallDelta>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize)]
struct ChatFunctionCallDelta {
    name: Option<String>,
    arguments: Option<String>,
}
