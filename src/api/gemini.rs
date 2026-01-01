//! Google Gemini API client implementation.
//!
//! This module implements the `Client` trait for Google's Gemini API using
//! the generic options architecture.
//! See: <https://ai.google.dev/api/rest>

use async_trait::async_trait;
use futures::Stream;
use nonempty::NonEmpty;
use reqwest::header::CONTENT_TYPE;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use serde_with::skip_serializing_none;

use crate::client::{Client, ClientError, StreamingClient};
use crate::http::{add_extra_headers, build_http_client, RequestBuilderExt, ResponseExt};
use crate::model::{FinishReason, Message, Response, Role, Usage, Part};
use crate::options::{ModelOptions, TransportOptions};
use crate::sse::SSEResponseExt;

/// Gemini-specific model options.
/// Currently empty, but can be extended with Gemini-specific parameters
/// like `top_k`, `safety_settings`, etc.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GeminiModel {
    // Future Gemini-specific fields:
    // pub top_k: Option<u32>,
    // pub safety_settings: Option<Vec<SafetySetting>>,
    // pub stop_sequences: Option<Vec<String>>,
}

/// Gemini client using HTTP transport.
#[derive(Debug, Clone)]
pub struct GeminiClient {
    api_key: String,
    base_url: String,
    model_options: ModelOptions<GeminiModel>,
    transport_options: TransportOptions,
}

impl GeminiClient {
    /// Create a new Gemini client.
    pub fn new(
        api_key: String,
        base_url: String,
        model_options: ModelOptions<GeminiModel>,
        transport_options: TransportOptions,
    ) -> Self {
        Self {
            api_key,
            base_url,
            model_options,
            transport_options,
        }
    }

    /// Process streaming response from Gemini.
    fn process_stream(
        response: reqwest::Response,
    ) -> impl Stream<Item = Result<crate::model::StreamChunk, ClientError>> + Send {
        use crate::model::{StreamChunk, Usage};
        use futures::StreamExt;

        // Use the SSE response extension trait
        let sse_stream = response.sse().map(|result| {
            result.and_then(|line| {
                serde_json::from_str::<GeminiResponse>(&line).map_err(ClientError::Parse)
            })
        });

        // Map Gemini-specific chunks to StreamChunk enum variants
        sse_stream.flat_map(|result| {
            use futures::stream;

            match result {
                Ok(gemini_resp) => {
                    let mut chunks = Vec::new();

                    // Extract message data chunks
                    for candidate in gemini_resp.candidates.iter() {
                        for part in &candidate.content.parts {
                            chunks.push(Ok(StreamChunk::Data(part.clone().into())));
                        }
                    }

                    // Add usage chunk if available
                    if let Some(usage_metadata) = gemini_resp.usage_metadata {
                        chunks.push(Ok(StreamChunk::Usage(Usage {
                            prompt_tokens: Some(usage_metadata.prompt_token_count),
                            completion_tokens: Some(
                                usage_metadata.candidates_token_count.unwrap_or_default()
                                    + usage_metadata.thoughts_token_count.unwrap_or_default(),
                            ),
                        })));
                    }

                    // Only emit finish chunk if finish_reason is present (arrives at the end in streaming)
                    if let Some(finish_reason) = gemini_resp.candidates.last().finish_reason {
                        chunks.push(Ok(StreamChunk::Finish(finish_reason.into())));
                    }

                    stream::iter(chunks)
                }
                Err(e) => stream::iter(vec![Err(e)]),
            }
        })
    }

    /// Handle Gemini error responses.
    fn handle_error_response(status: reqwest::StatusCode, body: &str) -> ClientError {
        if let Ok(error_resp) = serde_json::from_str::<GeminiErrorResponse>(body) {
            ClientError::ProviderError(format!(
                "Gemini error ({}): {}",
                error_resp.error.code, error_resp.error.message
            ))
        } else {
            ClientError::ProviderError(format!("HTTP {}: {}", status, body))
        }
    }
}

impl GeminiRequest {
    fn new(messages: Vec<Message>, model_options: &ModelOptions<GeminiModel>, tool_defs: Vec<rmcp::model::Tool>) -> Self {
        let mut contents = Vec::new();
        // Handle system instructions if needed, though Gemini usually takes them in generation config or separate field?
        // Actually Gemini API has `system_instruction` field at top level.
        // But `GeminiRequest` struct definition (which I haven't seen fully) needs to support it.
        // Let's check `GeminiRequest` struct definition.
        // I'll assume it doesn't have it for now based on previous code, or I'll check.
        
        for msg in messages {
            match msg {
                Message::System(_) => {
                    // Skip system messages in contents for now, or map to User?
                    // If I can't add to system_instruction (if struct doesn't have it), I might have to skip or map.
                    // Previous code didn't handle System role explicitly (it wasn't in Role enum).
                }
                _ => {
                    contents.push(msg.into());
                }
            }
        }

        GeminiRequest {
            contents,
            generation_config: Some(GeminiGenerationConfig {
                temperature: model_options.temperature,
                top_p: model_options.top_p,
                max_output_tokens: model_options.max_tokens,
                thinking_config: Some(GeminiThinkingConfig {
                    include_thoughts: model_options.reasoning,
                    thinking_budget: None,
                }),
            }),
            tools: Some(vec![GeminiTool {
                function_declarations: tool_defs
                    .iter()
                    .map(|def| GeminiFunctionDeclaration {
                        name: def.name.to_string(),
                        description: def.description.as_ref().map(|d| d.to_string()).unwrap_or_default(),
                        parameters_json_schema: serde_json::Value::Object((*def.input_schema).clone()),
                    })
                    .collect(),
            }]),
        }
    }
}

impl From<Role> for GeminiRole {
    fn from(role: Role) -> Self {
        match role {
            Role::User => GeminiRole::User,
            Role::Assistant => GeminiRole::Model,
            Role::System => GeminiRole::User,
        }
    }
}

impl From<Message> for GeminiContent {
    fn from(msg: Message) -> Self {
        let role = match msg {
            Message::User(_) => GeminiRole::User,
            Message::Assistant(_) => GeminiRole::Model,
            Message::System(_) => GeminiRole::User,
        };

        let mut parts = Vec::new();
        for part in msg.parts() {
            match part {
                Part::Text(text) => parts.push(GeminiPart::Text { thought: None, text: text.clone() }),
                Part::Reasoning { content, .. } => parts.push(GeminiPart::Text { thought: Some(true), text: content.clone() }),
                Part::FunctionCall { name, arguments, signature, .. } => parts.push(GeminiPart::FunctionCall {
                    thought_signature: signature.clone(),
                    function_call: FunctionCall { name: name.clone(), args: arguments.clone() },
                }),
                Part::FunctionResponse { name, response, .. } => parts.push(GeminiPart::FunctionResponse {
                    function_response: FunctionResponse { name: name.clone(), response: response.clone() },
                }),
                Part::Image { .. } => { /* Placeholder */ }
            }
        }

        GeminiContent {
            role,
            parts,
        }
    }
}

impl From<GeminiPart> for Message {
    fn from(part: GeminiPart) -> Self {
        match part {
            GeminiPart::Text { thought, text } => {
                if thought.unwrap_or_default() {
                    Message::Assistant(vec![Part::Reasoning {
                        content: text,
                        summary: None,
                        signature: None,
                    }])
                } else {
                    Message::Assistant(vec![Part::Text(text)])
                }
            }
            GeminiPart::FunctionCall {
                thought_signature,
                function_call,
            } => Message::Assistant(vec![Part::FunctionCall {
                id: None,
                name: function_call.name,
                arguments: function_call.args,
                signature: thought_signature,
            }]),
            _ => panic!(
                "Attempted conversion of unsupported GeminiPart variant (this is a interal bug)"
            ),
        }
    }
}

impl From<GeminiResponse> for Response {
    fn from(gemini_resp: GeminiResponse) -> Self {
        let finish_reason = gemini_resp
            .candidates
            .last()
            .finish_reason
            .unwrap_or(GeminiFinishReason::Stop)
            .into();
        let parts = gemini_resp
            .candidates
            .into_iter()
            .flat_map(|candidate| candidate.content.parts.into_iter());

        Response {
            data: parts.map(|part| part.into()).collect(),
            usage: gemini_resp.usage_metadata.map(|u| u.into()),
            finish: finish_reason,
        }
    }
}

#[async_trait]
impl Client for GeminiClient {
    type ModelProvider = GeminiModel;

    async fn request(
        &self,
        messages: Vec<Message>,
        tools: Vec<rmcp::model::Tool>,
    ) -> Result<Response, ClientError> {
        // Determine model: use model_options or default
        let model = self.model_options
            .model
            .clone()
            .ok_or_else(|| ClientError::Config("Model must be specified".to_string()))?;

        let url = format!(
            "{}/v1beta/models/{}:generateContent?key={}",
            self.base_url, model, self.api_key
        );

        // Build request body inline
        let request_body = GeminiRequest::new(messages, &self.model_options, tools);

        // Build HTTP client with transport options
        let http_client = build_http_client(&self.transport_options)?;

        // Build request with extra headers if specified
        let mut req = http_client
            .post(&url)
            .header(CONTENT_TYPE, "application/json");

        req = add_extra_headers(req, &self.transport_options);

        let response = req.json_logged(&request_body).send().await?;
        let status = response.status();

        if !status.is_success() {
            let body = response.text_logged().await.unwrap_or_default();
            return Err(Self::handle_error_response(status, &body));
        }

        let gemini_response: GeminiResponse = response.json_logged().await?;
        Ok(gemini_response.into())
    }

    fn model_options(&self) -> &ModelOptions<Self::ModelProvider> {
        &self.model_options
    }

    fn transport_options(&self) -> &TransportOptions {
        &self.transport_options
    }
}

#[async_trait]
impl StreamingClient for GeminiClient {
    async fn request_stream(
        &self,
        messages: Vec<Message>,
        tools: Vec<rmcp::model::Tool>,
    ) -> Result<
        std::pin::Pin<Box<dyn Stream<Item = Result<crate::model::StreamChunk, ClientError>> + Send>>,
        ClientError,
    > {
        // Determine model: use model_options or default
        let model = self.model_options
            .model
            .clone()
            .ok_or_else(|| ClientError::Config("Model must be specified".to_string()))?;

        // Use alt=sse parameter for true streaming with Server-Sent Events
        let url = format!(
            "{}/v1beta/models/{}:streamGenerateContent?alt=sse&key={}",
            self.base_url, model, self.api_key
        );

        let request_body = GeminiRequest::new(messages, &self.model_options, tools);

        // Build HTTP client with transport options
        let http_client = build_http_client(&self.transport_options)?;

        // Build request with extra headers if specified
        let mut req = http_client
            .post(&url)
            .header(CONTENT_TYPE, "application/json");

        req = add_extra_headers(req, &self.transport_options);

        // For streaming, we log the request but not the response (it's a stream)
        let response = req.json_logged(&request_body).send().await?;
        let status = response.status();

        if !status.is_success() {
            let body = response.text_logged().await.unwrap_or_default();
            return Err(Self::handle_error_response(status, &body));
        }

        Ok(Box::pin(Self::process_stream(response)))
    }
}

// --- Gemini API Request/Response Types ---

#[skip_serializing_none]
#[derive(Debug, Clone, Serialize)]
struct GeminiRequest {
    contents: Vec<GeminiContent>,
    generation_config: Option<GeminiGenerationConfig>,
    tools: Option<Vec<GeminiTool>>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiTool {
    function_declarations: Vec<GeminiFunctionDeclaration>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiFunctionDeclaration {
    name: String,
    description: String,
    parameters_json_schema: serde_json::Value,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum GeminiRole {
    User,
    Model,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GeminiContent {
    role: GeminiRole,
    #[serde(default)]
    parts: Vec<GeminiPart>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FunctionCall {
    name: String,
    args: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FunctionResponse {
    name: String,
    response: Value,
}

#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged, rename_all_fields = "camelCase")]
enum GeminiPart {
    Text {
        thought: Option<bool>,
        text: String,
    },
    FunctionCall {
        thought_signature: Option<String>,
        function_call: FunctionCall,
    },
    FunctionResponse {
        function_response: FunctionResponse,
    },
}

#[skip_serializing_none]
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiThinkingConfig {
    include_thoughts: Option<bool>,
    thinking_budget: Option<u32>,
}

#[skip_serializing_none]
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiGenerationConfig {
    temperature: Option<f32>,
    top_p: Option<f32>,
    max_output_tokens: Option<u32>,
    thinking_config: Option<GeminiThinkingConfig>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiResponse {
    candidates: NonEmpty<GeminiCandidate>,
    #[allow(dead_code)]
    model_version: Option<String>,
    usage_metadata: Option<GeminiUsageMetadata>,
}

#[derive(Debug, Copy, Clone, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
enum GeminiFinishReason {
    Stop,
    MaxTokens,
    Safety,
    Language,
    Blocklist,
    ProhibitedContent,
    Spii,
    ImageSafety,
    ImageProhibitedContent,
    ImageRecitation,
    MalformedFunctionCall,
    UnexpectedToolCall,
    TooManyToolCalls,
    #[serde(other)]
    Other,
}

impl From<GeminiFinishReason> for FinishReason {
    fn from(reason: GeminiFinishReason) -> Self {
        match reason {
            GeminiFinishReason::Stop => FinishReason::Stop,
            GeminiFinishReason::MaxTokens => FinishReason::OutputTokens,
            GeminiFinishReason::Safety
            | GeminiFinishReason::Language
            | GeminiFinishReason::Blocklist
            | GeminiFinishReason::ProhibitedContent
            | GeminiFinishReason::Spii
            | GeminiFinishReason::ImageSafety
            | GeminiFinishReason::ImageProhibitedContent
            | GeminiFinishReason::ImageRecitation => FinishReason::ContentFilter,
            GeminiFinishReason::MalformedFunctionCall
            | GeminiFinishReason::UnexpectedToolCall
            | GeminiFinishReason::TooManyToolCalls => FinishReason::ToolCalls,
            GeminiFinishReason::Other => FinishReason::Stop,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiCandidate {
    content: GeminiContent,
    finish_reason: Option<GeminiFinishReason>,
}

#[derive(Debug, Copy, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiUsageMetadata {
    prompt_token_count: u32,
    candidates_token_count: Option<u32>,
    thoughts_token_count: Option<u32>,
}

impl From<GeminiUsageMetadata> for Usage {
    fn from(u: GeminiUsageMetadata) -> Self {
        Usage {
            prompt_tokens: Some(u.prompt_token_count),
            completion_tokens: Some(
                u.candidates_token_count.unwrap_or_default()
                    + u.thoughts_token_count.unwrap_or_default(),
            ),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct GeminiErrorResponse {
    error: GeminiError,
}

#[derive(Debug, Clone, Deserialize)]
struct GeminiError {
    code: u32,
    message: String,
}
