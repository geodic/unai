//! Agent struct for automatic tool execution with LLM providers.

use crate::client::{Client, ClientError};
use crate::model::{FinishReason, Message, Part, Response, Usage};
use serde_json::json;
use std::collections::HashMap;
use tracing::{debug, info, warn};

use crate::mcp::MCPServer;

/// Agent that automatically executes tools in a loop.
///
/// Unlike the raw `Client`, an `Agent` handles tool execution automatically:
/// 1. Sends request with tool definitions from the MCP server (if configured)
/// 2. Receives response with potential function calls
/// 3. Executes tools automatically
/// 4. Adds results back to conversation
/// 5. Loops until no more function calls
///
/// # Example
/// ```ignore
/// use unia::agent::Agent;
/// use unia::providers::{Gemini, Provider};
/// use unia::model::{Message, Part};
///
/// let client = Gemini::create("api_key".to_string(), "gemini-3.0-pro".to_string());
/// let agent = Agent::new(client)
///     .with_server(weather_server);
///
/// let messages = vec![
///     Message::User(vec![
///         Part::Text { content: "What's the weather?".into(), finished: true }
///     ])
/// ];
///
/// let response = agent.chat(messages).await?;
/// ```
pub struct Agent<C: Client> {
    client: C,
    max_iterations: usize,
    server: Option<Box<dyn MCPServer>>,
}

impl<C: Client> Agent<C> {
    /// Create a new agent.
    ///
    /// # Arguments
    /// - `client`: The initialized client instance
    ///
    /// Tools are fetched from the configured MCP server if available.
    pub fn new(client: C) -> Self {
        Self {
            client,
            max_iterations: 10,
            server: None,
        }
    }

    /// Set the MCP server for the agent.
    pub fn with_server<S: MCPServer + 'static>(mut self, server: S) -> Self {
        self.server = Some(Box::new(server));
        self
    }

    /// Set the maximum number of iterations for the agentic loop.
    pub fn with_max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }

    /// Send a chat request with automatic tool execution.
    ///
    /// This method automatically handles the tool execution loop:
    /// - Sends request to LLM with tools from the MCP server (if configured)
    /// - Executes any tool calls
    /// - Continues until no more tool calls or max iterations reached
    ///
    /// # Arguments
    /// - `messages`: Conversation messages
    ///
    /// # Returns
    /// The response containing all new messages generated during the execution (including tool calls and results)
    pub async fn chat(&self, mut messages: Vec<Message>) -> Result<Response, ClientError> {
        debug!(
            "Starting agent chat loop with {} initial messages",
            messages.len()
        );

        let mut current_response = Response {
            data: Vec::new(),
            usage: Usage::default(),
            finish: FinishReason::Unfinished,
        };

        let (tools, tool_map) = if let Some(server) = &self.server {
            match server.list_tools().await {
                Ok(tools) => {
                    let map: HashMap<String, Option<String>> = tools
                        .iter()
                        .map(|t| (t.value.name.to_string(), t.server_id.clone()))
                        .collect();
                    (tools.into_iter().map(|t| t.value).collect(), map)
                }
                Err(e) => {
                    return Err(ClientError::ProviderError(format!(
                        "Failed to list tools from MCP server: {}",
                        e
                    )));
                }
            }
        } else {
            (Vec::new(), HashMap::new())
        };

        for iteration in 0..self.max_iterations {
            debug!("Agent iteration {}/{}", iteration + 1, self.max_iterations);

            let response = self.client.request(messages.clone(), tools.clone()).await?;
            current_response.usage += response.usage;
            current_response.finish = response.finish.clone();

            let mut tool_calls_executed = false;

            for msg in response.data {
                messages.push(msg.clone());
                current_response.data.push(msg.clone());

                for part in msg.parts() {
                    if let Part::FunctionCall {
                        id,
                        name,
                        arguments,
                        ..
                    } = part
                    {
                        tool_calls_executed = true;
                        info!("Tool call requested: {}", name);
                        debug!("Tool arguments: {}", arguments);

                        let server = self.server.as_ref().ok_or_else(|| {
                            ClientError::Config("No MCP server configured".to_string())
                        })?;
                        let server_id = tool_map.get(name).cloned().flatten();
                        let result = server
                            .call_tool(name.clone(), arguments.clone(), server_id)
                            .await;

                        let response_part = match result {
                            Ok(mut part) => {
                                info!("Tool {} executed successfully", name);
                                debug!("Tool result: {:?}", part);
                                if let Part::FunctionResponse {
                                    id: ref mut pid, ..
                                } = part
                                {
                                    *pid = id.clone();
                                }
                                part
                            }
                            Err(e) => {
                                warn!("Tool {} execution failed: {}", name, e);
                                Part::FunctionResponse {
                                    id: id.clone(),
                                    name: name.clone(),
                                    response: json!({ "error": format!("Error: {}", e) }),
                                    parts: vec![],
                                    finished: true,
                                }
                            }
                        };

                        let response_msg = Message::User(vec![response_part]);
                        messages.push(response_msg.clone());
                        current_response.data.push(response_msg);
                    }
                }
            }

            if !tool_calls_executed {
                debug!("No more function calls, agent loop complete");
                return Ok(current_response);
            }
        }

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
    /// - Sends streaming request to LLM with tools from the MCP server (if configured)
    /// - Executes any tool calls
    /// - Continues until no more tool calls or max iterations reached
    ///
    /// # Arguments
    /// - `messages`: Conversation messages
    ///
    /// # Returns
    /// A stream of chunks for the final response after all tool executions complete
    pub fn chat_stream<'a>(
        &'a self,
        mut messages: Vec<Message>,
    ) -> std::pin::Pin<Box<dyn futures::Stream<Item = Result<Response, ClientError>> + Send + 'a>>
    where
        C: crate::client::StreamingClient,
    {
        Box::pin(async_stream::try_stream! {
            debug!("Starting agent streaming chat loop");
            use futures::StreamExt;

            let mut current_response = Response {
                data: Vec::new(),
                usage: Usage::default(),
                finish: FinishReason::Unfinished,
            };

            let (tools, tool_map) = if let Some(server) = &self.server {
                match server.list_tools().await {
                    Ok(tools) => {
                        let map: HashMap<String, Option<String>> = tools
                            .iter()
                            .map(|t| (t.value.name.to_string(), t.server_id.clone()))
                            .collect();
                        (tools.into_iter().map(|t| t.value).collect(), map)
                    }
                    Err(e) => {
                        warn!("Failed to list tools from MCP server: {}", e);
                        (Vec::new(), HashMap::new())
                    }
                }
            } else {
                (Vec::new(), HashMap::new())
            };

            for iteration in 0..self.max_iterations {
                debug!(
                    "Agent streaming iteration {}/{}",
                    iteration + 1,
                    self.max_iterations
                );

                let mut stream = self.client.request_stream(messages.clone(), tools.clone()).await?;

                // Snapshot of state before this turn
                let base_data_len = current_response.data.len();
                let base_usage = current_response.usage.clone();

                while let Some(response_result) = stream.next().await {
                    let response = response_result?;

                    // Update current_response
                    // Truncate to base length to remove previous partials of this turn
                    current_response.data.truncate(base_data_len);
                    current_response.data.extend(response.data.clone());

                    current_response.usage = base_usage.clone();
                    current_response.usage += response.usage;
                    current_response.finish = response.finish;

                    yield current_response.clone();
                }

                // After stream, current_response contains the full assistant message for this turn.
                // Update messages history
                if current_response.data.len() > base_data_len {
                     // The new messages added in this turn
                     for i in base_data_len..current_response.data.len() {
                         messages.push(current_response.data[i].clone());
                     }
                }

                // Check for tool calls
                let mut tool_calls_executed = false;
                let mut tool_responses = Vec::new();

                // We only check the LAST message for tool calls, which should be the assistant's message
                if let Some(msg) = current_response.data.last() {
                    for part in msg.parts() {
                        if let Part::FunctionCall { id, name, arguments, finished, .. } = part {
                            if *finished {
                                tool_calls_executed = true;
                                info!("Executing tool: {}", name);

                                let server = self.server.as_ref().ok_or_else(|| ClientError::Config("No MCP server configured".to_string()))?;
                                let server_id = tool_map.get(name).cloned().flatten();
                                let result = server
                                    .call_tool(name.clone(), arguments.clone(), server_id)
                                    .await;

                                let response_part = match result {
                                    Ok(mut part) => {
                                        if let Part::FunctionResponse { id: ref mut pid, .. } = part {
                                            *pid = id.clone();
                                        }
                                        part
                                    }
                                    Err(e) => {
                                        Part::FunctionResponse {
                                            id: id.clone(),
                                            name: name.clone(),
                                            response: json!({ "error": format!("Error: {}", e) }),
                                            parts: vec![],
                                            finished: true,
                                        }
                                    },
                                };
                                tool_responses.push(response_part);
                            }
                        }
                    }
                }

                if tool_calls_executed {
                    let tool_msg = Message::User(tool_responses);
                    messages.push(tool_msg.clone());
                    current_response.data.push(tool_msg);

                    yield current_response.clone();
                } else {
                    // No tool calls, we are done
                    return;
                }
            }

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
