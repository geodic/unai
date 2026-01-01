//! Agent struct for automatic tool execution with LLM providers.

use crate::client::{Client, ClientError};
use crate::model::{Message, Response, Part, Usage};
use serde_json::json;
use tracing::{debug, info, warn};
use rmcp::service::{RunningService, RoleClient};
use rmcp::ClientHandler;
use rmcp::model::{Tool, CallToolRequestParam, RawContent, Prompt, Resource, GetPromptRequestParam, ReadResourceRequestParam, ResourceContents};
use async_trait::async_trait;
use std::ops::Deref;

use crate::mcp::McpServer;

/// Agent that automatically executes tools in a loop.
///
/// Unlike the raw `Client`, an `Agent` handles tool execution automatically:
/// 1. Sends request with tool definitions from Context
/// 2. Receives response with potential function calls
/// 3. Executes tools automatically
/// 4. Adds results back to conversation
/// 5. Loops until no more function calls
///
/// # Example
/// ```ignore
/// let client = GeminiProvider::create(api_key);
/// let agent = Agent::new(client)
///     .with_mcp_server(weather_server);
///
/// let messages = vec![Message::Text { role: Role::User, content: "Hello".into() }];
///
/// let response = agent.chat(messages, vec![]).await?;
/// ```
pub struct Agent<C: Client> {
    client: C,
    max_iterations: usize,
    server: Option<Box<dyn McpServer>>,
}

impl<C: Client> Agent<C> {
    /// Create a new agent.
    ///
    /// # Arguments
    /// - `client`: The initialized client instance
    ///
    /// Tools are passed via Context in chat/chat_stream methods.
    pub fn new(client: C) -> Self {
        Self {
            client,
            max_iterations: 10,
            server: None,
        }
    }

    /// Set the MCP server for the agent.
    pub fn with_server<S: McpServer + 'static>(mut self, server: S) -> Self {
        self.server = Some(Box::new(server));
        self
    }

    /// Set the maximum number of iterations for the agentic loop.
    pub fn with_max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }

    /// List available prompts from the MCP server.
    pub async fn list_prompts(&self) -> Result<Vec<Prompt>, String> {
        if let Some(server) = &self.server {
            server.list_prompts().await
        } else {
            Ok(Vec::new())
        }
    }

    /// Get a prompt from the MCP server.
    pub async fn get_prompt(&self, name: String, args: Option<serde_json::Map<String, serde_json::Value>>) -> Result<rmcp::model::GetPromptResult, String> {
        if let Some(server) = &self.server {
            server.get_prompt(name, args).await
        } else {
            Err(format!("No MCP server configured"))
        }
    }

    /// List available resources from the MCP server.
    pub async fn list_resources(&self) -> Result<Vec<Resource>, String> {
        if let Some(server) = &self.server {
            server.list_resources().await
        } else {
            Ok(Vec::new())
        }
    }

    /// Read a resource from the MCP server.
    pub async fn read_resource(&self, uri: String) -> Result<rmcp::model::ReadResourceResult, String> {
        if let Some(server) = &self.server {
            server.read_resource(uri).await
        } else {
            Err(format!("No MCP server configured"))
        }
    }

    /// Prepare a context from an MCP prompt.
    ///
    /// This fetches the prompt messages and creates a new Context.
    /// You can then add resources or other messages to this context.
    pub async fn prompt(&self, name: String, args: Option<serde_json::Map<String, serde_json::Value>>) -> Result<Vec<Message>, String> {
        let prompt_result = self.get_prompt(name, args).await?;
        
        let mut messages = Vec::new();
        for msg in prompt_result.messages {
            let role = match msg.role {
                rmcp::model::PromptMessageRole::User => crate::model::Role::User,
                rmcp::model::PromptMessageRole::Assistant => crate::model::Role::Assistant,
            };
            
            let content = match msg.content {
                rmcp::model::PromptMessageContent::Text { text } => text,
                rmcp::model::PromptMessageContent::Image { .. } => {
                    warn!("Image content in prompt not supported yet, skipping");
                    continue;
                }
                rmcp::model::PromptMessageContent::Resource { .. } => {
                    warn!("Resource content in prompt not supported yet, skipping");
                    continue;
                }
                rmcp::model::PromptMessageContent::ResourceLink { .. } => {
                    warn!("Resource link in prompt not supported yet, skipping");
                    continue;
                }
            };

            let part = Part::Text(content);
            let message = match role {
                crate::model::Role::User => Message::User(vec![part]),
                crate::model::Role::Assistant => Message::Assistant(vec![part]),
                crate::model::Role::System => Message::System(vec![part]),
            };

            messages.push(message);
        }

        Ok(messages)
    }

    // --- Helper Methods ---

    async fn load_resources(&self, messages: &mut Vec<Message>, resources: &[String]) {
        for uri in resources {
            match self.read_resource(uri.clone()).await {
                Ok(result) => {
                    for content in result.contents {
                        match content {
                            ResourceContents::TextResourceContents { uri, text, .. } => {
                                messages.push(Message::User(vec![Part::Text(format!("Resource {}:\n{}", uri, text))]));
                            }
                            ResourceContents::BlobResourceContents { uri, .. } => {
                                warn!("Blob resource {} not supported yet, skipping", uri);
                            }
                        }
                    }
                }
                Err(e) => warn!("Failed to read resource {}: {}", uri, e),
            }
        }
    }

    async fn get_tools(&self) -> Vec<Tool> {
        if let Some(server) = &self.server {
            match server.list_tools().await {
                Ok(tools) => tools,
                Err(e) => {
                    warn!("Failed to list tools from MCP server: {}", e);
                    Vec::new()
                }
            }
        } else {
            Vec::new()
        }
    }

    async fn execute_tool(&self, name: &str, arguments: serde_json::Value) -> Result<serde_json::Value, String> {
        if let Some(server) = &self.server {
            server.call_tool(name.to_string(), arguments).await
        } else {
            Err(format!("No MCP server configured"))
        }
    }

    /// Send a chat request with automatic tool execution.
    ///
    /// This method automatically handles the tool execution loop:
    /// - Sends request to LLM with tools from Context
    /// - Executes any tool calls
    /// - Continues until no more tool calls or max iterations reached
    ///
    /// # Arguments
    /// - `messages`: Conversation messages
    /// - `resources`: List of resource URIs to fetch and add to context
    ///
    /// # Returns
    /// The response containing all new messages generated during the execution (including tool calls and results)
    pub async fn chat(&self, mut messages: Vec<Message>, resources: Vec<String>) -> Result<Response, ClientError> {
        debug!(
            "Starting agent chat loop with {} initial messages",
            messages.len()
        );

        let mut new_messages = Vec::new();

        // Fetch and add resources
        let start_len = messages.len();
        self.load_resources(&mut messages, &resources).await;
        for i in start_len..messages.len() {
            new_messages.push(messages[i].clone());
        }

        let tools = self.get_tools().await;
        let mut total_usage = Usage { prompt_tokens: Some(0), completion_tokens: Some(0) };

        for iteration in 0..self.max_iterations {
            debug!("Agent iteration {}/{}", iteration + 1, self.max_iterations);

            // Send request
            let response = self.client.request(messages.clone(), tools.clone()).await?;
            if let Some(usage) = &response.usage {
                total_usage += usage.clone();
            }

            let mut tool_calls_executed = false;

            // Process response messages and execute tools if any
            for msg in response.data {
                messages.push(msg.clone());
                new_messages.push(msg.clone());

                for part in msg.parts() {
                    if let Part::FunctionCall { id, name, arguments, .. } = part {
                        tool_calls_executed = true;
                        info!("Tool call requested: {}", name);
                        debug!("Tool arguments: {}", arguments);

                        // Execute tool
                        let result = self.execute_tool(&name, arguments.clone()).await;

                        let response_part = match result {
                            Ok(result) => {
                                info!("Tool {} executed successfully", name);
                                debug!("Tool result: {}", result);
                                Part::FunctionResponse {
                                    id: id.clone(),
                                    name: name.clone(),
                                    response: result,
                                }
                            }
                            Err(e) => {
                                warn!("Tool {} execution failed: {}", name, e);
                                Part::FunctionResponse {
                                    id: id.clone(),
                                    name: name.clone(),
                                    response: json!({"error": e.to_string()}),
                                }
                            }
                        };
                        
                        let response_msg = Message::User(vec![response_part]);
                        messages.push(response_msg.clone());
                        new_messages.push(response_msg);
                    }
                }
            }

            // If no function calls, we're done
            if !tool_calls_executed {
                debug!("No more function calls, agent loop complete");
                return Ok(Response {
                    data: new_messages,
                    usage: Some(total_usage),
                    finish: response.finish,
                });
            }
        }

        // Max iterations reached
        warn!(
            "Max iterations ({}) reached in agent loop",
            self.max_iterations
        );
        Err(ClientError::Config(
            "Max iterations reached in agent loop".to_string(),
        ))
    }

    /// Send a streaming chat request with automatic tool execution.
    ///
    /// This method automatically handles the tool execution loop with streaming:
    /// - Sends streaming request to LLM with tools from Context
    /// - Executes any tool calls
    /// - Continues until no more tool calls or max iterations reached
    ///
    /// # Arguments
    /// - `messages`: Conversation messages
    /// - `resources`: List of resource URIs to fetch and add to context
    ///
    /// # Returns
    /// A stream of chunks for the final response after all tool executions complete
    pub fn chat_stream<'a>(
        &'a self,
        mut messages: Vec<Message>,
        resources: Vec<String>,
    ) -> std::pin::Pin<
        Box<dyn futures::Stream<Item = Result<crate::stream::StreamChunk, ClientError>> + Send + 'a>,
    >
    where
        C: crate::client::StreamingClient,
    {
        Box::pin(async_stream::try_stream! {
            debug!(
                "Starting agent streaming chat loop with {} initial messages",
                messages.len()
            );

            use futures::StreamExt;

            // Fetch and add resources
            let start_len = messages.len();
            self.load_resources(&mut messages, &resources).await;
            for i in start_len..messages.len() {
                yield crate::stream::StreamChunk::Data(messages[i].clone());
            }

            let tools = self.get_tools().await;
            
            let mut total_usage = Usage { prompt_tokens: Some(0), completion_tokens: Some(0) };

            // Execute tool loop until we get a non-tool response
            for iteration in 0..self.max_iterations {
                debug!(
                    "Agent streaming iteration {}/{}",
                    iteration + 1,
                    self.max_iterations
                );

                // Stream the request
                let mut stream = self.client.request_stream(messages.clone(), tools.clone()).await?;
                
                let mut accumulated_message = Message::Assistant(Vec::new());
                let mut finish_reason = None;
                let request_base_usage = total_usage.clone();

                while let Some(chunk_result) = stream.next().await {
                    let chunk = chunk_result?;
                    
                    match chunk {
                        crate::stream::StreamChunk::Data(message) => {
                            yield crate::stream::StreamChunk::Data(message.clone());
                            
                            // Merge message into accumulated_message
                            let target_parts = accumulated_message.parts_mut();
                            for part in message.parts() {
                                let mut merged = false;
                                if let Some(Part::Text(last_text)) = target_parts.last_mut() {
                                    if let Part::Text(new_text) = part {
                                        last_text.push_str(new_text);
                                        merged = true;
                                    }
                                }
                                
                                if !merged {
                                    if let Some(Part::FunctionCall { arguments: last_args, .. }) = target_parts.last_mut() {
                                        if let Part::FunctionCall { arguments: new_args, .. } = part {
                                            if let (serde_json::Value::String(last_s), serde_json::Value::String(new_s)) = (last_args, new_args) {
                                                last_s.push_str(new_s);
                                                merged = true;
                                            }
                                        }
                                    }
                                }

                                if !merged {
                                    target_parts.push(part.clone());
                                }
                            }
                        }
                        crate::stream::StreamChunk::Usage(usage) => {
                            total_usage = request_base_usage.clone() + usage;
                            yield crate::stream::StreamChunk::Usage(total_usage.clone());
                        }
                        crate::stream::StreamChunk::Finish(reason) => {
                            finish_reason = Some(reason);
                        }
                    }
                }
                
                // Push accumulated message to history
                messages.push(accumulated_message.clone());

                // Check for tool calls in the accumulated message
                let mut tool_calls_executed = false;
                for part in accumulated_message.parts() {
                    if let Part::FunctionCall { id, name, arguments, .. } = part {
                        tool_calls_executed = true;
                        info!("Tool call requested (streaming): {}", name);
                        debug!("Tool arguments: {}", arguments);

                        // Execute tool
                        let args_value = match arguments {
                            serde_json::Value::String(s) => serde_json::from_str(s).unwrap_or(json!({})),
                            _ => arguments.clone(),
                        };
                        let result = self.execute_tool(&name, args_value).await;
                        
                        let response_part = match result {
                            Ok(response) => {
                                info!("Tool {} executed successfully (streaming)", name);
                                Part::FunctionResponse {
                                    id: id.clone(),
                                    name: name.clone(),
                                    response,
                                }
                            }
                            Err(e) => {
                                warn!("Tool {} execution failed (streaming): {}", name, e);
                                Part::FunctionResponse {
                                    id: id.clone(),
                                    name: name.clone(),
                                    response: json!({"error": e.to_string()}),
                                }
                            }
                        };
                        
                        let response_msg = Message::User(vec![response_part]);
                        messages.push(response_msg.clone());
                        yield crate::stream::StreamChunk::Data(response_msg);
                    }
                }

                if !tool_calls_executed {
                    // No function calls - we are done
                    if let Some(reason) = finish_reason {
                        yield crate::stream::StreamChunk::Finish(reason);
                    }
                    return;
                }
            }

            // Max iterations reached
            warn!(
                "Max iterations ({}) reached in streaming agent loop",
                self.max_iterations
            );
            Err(ClientError::Config(
                "Max iterations reached in agent loop".to_string(),
            ))?;
        })
    }
}
