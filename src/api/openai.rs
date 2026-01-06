//! OpenAI Chat Completions API client implementation.

use async_trait::async_trait;
use futures::{Stream, StreamExt};
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use serde_with::skip_serializing_none;
use std::collections::HashMap;
use std::pin::Pin;

use crate::client::{Client, ClientError, StreamingClient};
use crate::http::{add_extra_headers, build_http_client, RequestBuilderExt, ResponseExt};
use crate::model::{FinishReason, MediaType, Message, Part, Response, Usage};
use crate::options::{ModelOptions, TransportOptions};
use crate::sse::SSEResponseExt;

/// Trait for models compatible with OpenAI's Chat Completions API.
pub trait OpenAICompatibleModel:
    Send + Sync + Default + Serialize + for<'de> Deserialize<'de> + Clone
{
}

/// Generic client for OpenAI-compatible Chat Completions APIs.
#[derive(Debug, Clone)]
pub struct OpenAIClient<M> {
    api_key: String,
    base_url: String,
    model_options: ModelOptions<M>,
    transport_options: TransportOptions,
}

impl<M: OpenAICompatibleModel> OpenAIClient<M> {
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

    fn handle_error_response(status: reqwest::StatusCode, body: &str) -> ClientError {
        if let Ok(error_resp) = serde_json::from_str::<OpenAIErrorResponse>(body) {
            ClientError::ProviderError(format!(
                "OpenAI error ({}): {}",
                error_resp.error.error_type, error_resp.error.message
            ))
        } else {
            ClientError::ProviderError(format!("HTTP {}: {}", status, body))
        }
    }

    fn build_request(
        &self,
        messages: Vec<Message>,
        tools: Vec<rmcp::model::Tool>,
        stream: bool,
    ) -> Result<reqwest::RequestBuilder, ClientError> {
        let url = format!("{}/chat/completions", self.base_url);

        let model = self.model_options.model.clone();

        let request_body = OpenAIRequest::new(messages, &self.model_options, model, tools, stream);

        let http_client = build_http_client(&self.transport_options)?;

        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", self.api_key))
                .map_err(|_| ClientError::Config("Invalid API key".to_string()))?,
        );

        let mut req = http_client.post(&url).headers(headers);
        req = add_extra_headers(req, &self.transport_options);

        Ok(req.json_logged(&request_body))
    }
}

#[async_trait]
impl<M: OpenAICompatibleModel> Client for OpenAIClient<M> {
    type ModelProvider = M;

    async fn request(
        &self,
        messages: Vec<Message>,
        tools: Vec<rmcp::model::Tool>,
    ) -> Result<Response, ClientError> {
        let req = self.build_request(messages, tools, false)?;

        let response = req.send().await?;
        let status = response.status();

        if !status.is_success() {
            let body = response.text_logged().await.unwrap_or_default();
            return Err(Self::handle_error_response(status, &body));
        }

        let openai_response: OpenAIResponse = response.json_logged().await?;
        Ok(openai_response.into())
    }

    fn model_options(&self) -> &ModelOptions<Self::ModelProvider> {
        &self.model_options
    }

    fn transport_options(&self) -> &TransportOptions {
        &self.transport_options
    }
}

#[async_trait]
impl<M: OpenAICompatibleModel> StreamingClient for OpenAIClient<M> {
    async fn request_stream(
        &self,
        messages: Vec<Message>,
        tools: Vec<rmcp::model::Tool>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Response, ClientError>> + Send>>, ClientError>
    {
        let req = self.build_request(messages, tools, true)?;
        let response = req.send().await?;
        let status = response.status();

        if !status.is_success() {
            let body = response.text_logged().await.unwrap_or_default();
            return Err(Self::handle_error_response(status, &body));
        }

        Ok(Box::pin(OpenAIStream::create(response)))
    }
}

// --- Streaming Implementation ---

struct OpenAIStream;

impl OpenAIStream {
    fn create(
        response: reqwest::Response,
    ) -> impl Stream<Item = Result<Response, ClientError>> + Send {
        let sse_stream = response.sse();

        Box::pin(async_stream::try_stream! {
            let mut stream = Box::pin(sse_stream);
            let mut current_response = Response {
                data: vec![Message::Assistant(vec![])],
                usage: Usage::default(),
                finish: FinishReason::Unfinished,
            };

            let mut tool_index_map: HashMap<u32, usize> = HashMap::new();
            let mut current_text_part_index: Option<usize> = None;

            while let Some(event_result) = stream.next().await {
                let event_str = event_result?;

                let chunk_result: OpenAIStreamChunk = serde_json::from_str(&event_str)
                    .map_err(|e| ClientError::ProviderError(format!("JSON parse error: {} | Input: {}", e, event_str)))?;

                if let Some(usage) = chunk_result.usage {
                    current_response.usage.prompt_tokens = Some(usage.prompt_tokens);
                    current_response.usage.completion_tokens = Some(usage.completion_tokens);
                }

                for choice in chunk_result.choices {
                    let parts = current_response.data[0].parts_mut();

                    if let Some(delta) = choice.delta {
                        if let Some(delta_content) = delta.content {
                            if let Some(idx) = current_text_part_index {
                                if let Some(Part::Text { content, .. }) = parts.get_mut(idx) {
                                    content.push_str(&delta_content);
                                }
                            } else {
                                parts.push(Part::Text { content: delta_content, finished: false });
                                current_text_part_index = Some(parts.len() - 1);
                            }
                        }

                        if let Some(tool_calls) = delta.tool_calls {
                            for tool_call in tool_calls {
                                let idx = *tool_index_map.entry(tool_call.index).or_insert_with(|| {
                                    parts.push(Part::FunctionCall {
                                        id: None,
                                        name: String::new(),
                                        arguments: Value::String(String::new()),
                                        signature: None,
                                        finished: false,
                                    });
                                    parts.len() - 1
                                });

                                if let Some(Part::FunctionCall { id: p_id, name: p_name, arguments: p_args, .. }) = parts.get_mut(idx) {
                                    if let Some(id) = tool_call.id {
                                        *p_id = Some(id);
                                    }
                                    if let Some(function) = tool_call.function {
                                        if let Some(name) = function.name {
                                            p_name.push_str(&name);
                                        }
                                        if let Some(args) = function.arguments {
                                            if let Value::String(arg_str) = p_args {
                                                arg_str.push_str(&args);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    if let Some(finish_reason) = choice.finish_reason {
                        for part in parts.iter_mut() {
                            match part {
                                Part::Text { finished, .. } => *finished = true,
                                Part::Reasoning { finished, .. } => *finished = true,
                                Part::FunctionCall { finished, arguments, .. } => {
                                    *finished = true;
                                    if let Value::String(json_str) = arguments {
                                        if let Ok(json_val) = serde_json::from_str(json_str) {
                                            *arguments = json_val;
                                        } else {
                                            *arguments = json!({});
                                        }
                                    }
                                },
                                Part::FunctionResponse { finished, .. } => *finished = true,
                                Part::Media { finished, .. } => *finished = true,
                            }
                        }

                        current_response.finish = match finish_reason.as_str() {
                            "stop" => FinishReason::Stop,
                            "length" => FinishReason::OutputTokens,
                            "tool_calls" => FinishReason::ToolCalls,
                            "content_filter" => FinishReason::ContentFilter,
                            _ => FinishReason::Stop,
                        };
                    }
                }

                yield current_response.clone();
            }
        })
    }
}

// --- Request Types ---

#[skip_serializing_none]
#[derive(Debug, Serialize)]
struct OpenAIRequest<M> {
    model: String,
    messages: Vec<OpenAIMessage>,
    max_tokens: Option<u32>,
    #[serde(rename = "max_completion_tokens")]
    max_completion_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<OpenAITool>,
    #[serde(flatten)]
    provider_options: M,
}

#[derive(Debug, Serialize)]
struct OpenAIMessage {
    role: String,
    content: OpenAIContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tool_calls: Vec<OpenAIToolCall>,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum OpenAIContent {
    Text(String),
    Parts(Vec<OpenAIContentPart>),
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum OpenAIContentPart {
    Text { text: String },
    ImageUrl { image_url: OpenAIImageUrl },
    File { file: OpenAIFileContent },
}

#[derive(Debug, Serialize)]
struct OpenAIFileContent {
    #[serde(skip_serializing_if = "Option::is_none")]
    file_data: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    file_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    filename: Option<String>,
}

#[derive(Debug, Serialize)]
struct OpenAIImageUrl {
    url: String,
}

#[derive(Debug, Serialize)]
struct OpenAITool {
    #[serde(rename = "type")]
    tool_type: String,
    function: OpenAIFunction,
}

#[derive(Debug, Serialize)]
struct OpenAIFunction {
    name: String,
    description: Option<String>,
    parameters: Value,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenAIToolCall {
    id: String,
    #[serde(rename = "type")]
    call_type: String,
    function: OpenAIFunctionCall,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenAIFunctionCall {
    name: String,
    arguments: String,
}

impl<M: OpenAICompatibleModel> OpenAIRequest<M> {
    fn new(
        messages_in: Vec<Message>,
        model_options: &ModelOptions<M>,
        model: String,
        tool_defs: Vec<rmcp::model::Tool>,
        stream: bool,
    ) -> Self {
        let mut messages = Vec::new();

        if let Some(system) = &model_options.system {
            messages.push(OpenAIMessage {
                role: "system".to_string(),
                content: OpenAIContent::Text(system.clone()),
                name: None,
                tool_call_id: None,
                tool_calls: Vec::new(),
            });
        }

        for msg in messages_in {
            let role = match msg {
                Message::User(_) => "user",
                Message::Assistant(_) => "assistant",
            };

            let mut content_parts = Vec::new();
            let mut tool_calls = Vec::new();
            let mut tool_call_id = None;
            let name = None;

            for part in msg.parts() {
                match part {
                    Part::Text { content: t, .. } => {
                        content_parts.push(OpenAIContentPart::Text { text: t.clone() })
                    }
                    Part::Media {
                        media_type: MediaType::Image,
                        data,
                        mime_type,
                        ..
                    } => {
                        let anchor_text = part.anchor_media();
                        content_parts.push(OpenAIContentPart::Text { text: anchor_text });
                        content_parts.push(OpenAIContentPart::ImageUrl {
                            image_url: OpenAIImageUrl {
                                url: format!("data:{};base64,{}", mime_type, data),
                            },
                        });
                    }
                    Part::Media { data, uri, .. } => {
                        let anchor_text = part.anchor_media();
                        content_parts.push(OpenAIContentPart::Text { text: anchor_text });
                        content_parts.push(OpenAIContentPart::File {
                            file: OpenAIFileContent {
                                file_data: Some(data.clone()),
                                file_id: None,
                                filename: uri.clone(),
                            },
                        });
                    }
                    Part::FunctionCall {
                        id: Some(call_id),
                        name: fn_name,
                        arguments,
                        ..
                    } => {
                        tool_calls.push(OpenAIToolCall {
                            id: call_id.clone(),
                            call_type: "function".to_string(),
                            function: OpenAIFunctionCall {
                                name: fn_name.clone(),
                                arguments: arguments.to_string(),
                            },
                        });
                    }
                    Part::FunctionResponse {
                        id: Some(call_id),
                        response,
                        parts,
                        ..
                    } => {
                        tool_call_id = Some(call_id.clone());

                        let mut content_str = String::new();

                        if response != &serde_json::json!({}) {
                            content_str.push_str(&response.to_string());
                        }

                        for part in parts {
                            if let Part::Media {
                                media_type,
                                mime_type,
                                ..
                            } = part
                            {
                                let anchor_text = part.anchor_media();
                                content_str.push_str(&format!("\n{}", anchor_text));

                                match media_type {
                                    MediaType::Image => content_str.push_str("\n[Image Content]"),
                                    _ => content_str.push_str(&format!("\n[File: {}]", mime_type)),
                                }
                            }
                        }

                        content_parts.push(OpenAIContentPart::Text { text: content_str });
                    }
                    _ => {}
                }
            }

            let final_role = if tool_call_id.is_some() { "tool" } else { role };

            let content = if content_parts.len() == 1 {
                if let OpenAIContentPart::Text { text } = &content_parts[0] {
                    OpenAIContent::Text(text.clone())
                } else {
                    OpenAIContent::Parts(content_parts)
                }
            } else if !content_parts.is_empty() {
                OpenAIContent::Parts(content_parts)
            } else {
                OpenAIContent::Text(String::new())
            };

            messages.push(OpenAIMessage {
                role: final_role.to_string(),
                content,
                name,
                tool_call_id,
                tool_calls,
            });
        }

        let tools = tool_defs
            .into_iter()
            .map(|t| OpenAITool {
                tool_type: "function".to_string(),
                function: OpenAIFunction {
                    name: t.name.into_owned(),
                    description: t.description.map(|d| d.into_owned()),
                    parameters: Value::Object((*t.input_schema).clone()),
                },
            })
            .collect();

        let is_reasoning_model = model.starts_with("o1") || model.starts_with("o3");
        let (max_tokens, max_completion_tokens) = if is_reasoning_model {
            (None, model_options.max_tokens)
        } else {
            (model_options.max_tokens, None)
        };

        OpenAIRequest {
            model,
            messages,
            max_tokens,
            max_completion_tokens,
            temperature: model_options.temperature,
            top_p: model_options.top_p,
            stream: if stream { Some(true) } else { None },
            tools,
            provider_options: model_options.provider.clone(),
        }
    }
}

// --- Response Types ---

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OpenAIResponse {
    id: String,
    choices: Vec<OpenAIChoice>,
    usage: Option<OpenAIUsage>,
}

#[derive(Debug, Deserialize)]
struct OpenAIChoice {
    message: OpenAIResponseMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OpenAIResponseMessage {
    role: String,
    content: Option<String>,
    tool_calls: Option<Vec<OpenAIToolCall>>,
}

#[derive(Debug, Deserialize)]
struct OpenAIUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct OpenAIErrorResponse {
    error: OpenAIError,
}

#[derive(Debug, Deserialize)]
struct OpenAIError {
    #[serde(rename = "type")]
    error_type: String,
    message: String,
}

impl From<OpenAIResponse> for Response {
    fn from(resp: OpenAIResponse) -> Self {
        let mut parts = Vec::new();
        let mut finish_reason = FinishReason::Stop;

        if let Some(choice) = resp.choices.first() {
            if let Some(content) = &choice.message.content {
                parts.push(Part::Text {
                    content: content.clone(),
                    finished: true,
                });
            }
            if let Some(tool_calls) = &choice.message.tool_calls {
                for tool_call in tool_calls {
                    parts.push(Part::FunctionCall {
                        id: Some(tool_call.id.clone()),
                        name: tool_call.function.name.clone(),
                        arguments: serde_json::from_str(&tool_call.function.arguments)
                            .unwrap_or(Value::Null),
                        signature: None,
                        finished: true,
                    });
                }
            }

            if let Some(reason) = &choice.finish_reason {
                finish_reason = match reason.as_str() {
                    "stop" => FinishReason::Stop,
                    "length" => FinishReason::OutputTokens,
                    "tool_calls" => FinishReason::ToolCalls,
                    "content_filter" => FinishReason::ContentFilter,
                    _ => FinishReason::Stop,
                };
            }
        }

        let usage = resp
            .usage
            .map(|u| Usage {
                prompt_tokens: Some(u.prompt_tokens),
                completion_tokens: Some(u.completion_tokens),
            })
            .unwrap_or_default();

        Response {
            data: vec![Message::Assistant(parts)],
            usage,
            finish: finish_reason,
        }
    }
}

// --- Stream Types ---

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OpenAIStreamChunk {
    id: String,
    choices: Vec<OpenAIStreamChoice>,
    usage: Option<OpenAIUsage>,
}

#[derive(Debug, Deserialize)]
struct OpenAIStreamChoice {
    delta: Option<OpenAIDelta>,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAIDelta {
    content: Option<String>,
    tool_calls: Option<Vec<OpenAIStreamToolCall>>,
}

#[derive(Debug, Deserialize)]
struct OpenAIStreamToolCall {
    index: u32,
    id: Option<String>,
    function: Option<OpenAIStreamFunction>,
}

#[derive(Debug, Deserialize)]
struct OpenAIStreamFunction {
    name: Option<String>,
    arguments: Option<String>,
}
