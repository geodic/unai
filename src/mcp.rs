use async_trait::async_trait;
use rmcp::model::{Tool, Prompt, Resource, GetPromptResult, ReadResourceResult, CallToolRequestParam, RawContent, GetPromptRequestParam, ReadResourceRequestParam};
use rmcp::service::{RunningService, RoleClient};
use rmcp::ClientHandler;
use serde_json::Value;
use std::ops::Deref;

/// Trait for MCP servers that can be used by the Agent.
#[async_trait]
pub trait McpServer: Send + Sync {
    /// List available tools.
    async fn list_tools(&self) -> Result<Vec<Tool>, String>;

    /// Execute a tool.
    async fn call_tool(&self, name: String, args: Value) -> Result<Value, String>;

    /// List available prompts.
    async fn list_prompts(&self) -> Result<Vec<Prompt>, String>;

    /// Get a prompt.
    async fn get_prompt(&self, name: String, args: Option<serde_json::Map<String, Value>>) -> Result<GetPromptResult, String>;

    /// List available resources.
    async fn list_resources(&self) -> Result<Vec<Resource>, String>;

    /// Read a resource.
    async fn read_resource(&self, uri: String) -> Result<ReadResourceResult, String>;
}

#[async_trait]
impl<S> McpServer for RunningService<RoleClient, S>
where
    S: ClientHandler + Send + Sync + 'static,
{
    async fn list_tools(&self) -> Result<Vec<Tool>, String> {
        let result = self.deref()
            .list_tools(None)
            .await
            .map_err(|e| e.to_string())?;
        Ok(result.tools)
    }

    async fn call_tool(&self, name: String, args: serde_json::Value) -> Result<serde_json::Value, String> {
        let params = CallToolRequestParam {
            name: name.into(),
            arguments: args.as_object().cloned(),
        };
        
        let result = self.deref().call_tool(params)
            .await
            .map_err(|e| e.to_string())?;

        // If structured content is available, use it directly
        if let Some(structured) = result.structured_content {
            if structured.is_object() {
                return Ok(structured);
            } else {
                return Ok(serde_json::json!({ "result": structured }));
            }
        }

        let mut content_values = Vec::new();
        for content in result.content {
            match content.raw {
                RawContent::Text(text_content) => {
                    // Wrap text content in a JSON object
                    content_values.push(serde_json::json!({
                        "content": text_content.text
                    }));
                }
                RawContent::Image(image_content) => {
                    if let Ok(val) = serde_json::to_value(&image_content) {
                        content_values.push(serde_json::json!({
                            "type": "image",
                            "content": val,
                            "annotations": content.annotations
                        }));
                    }
                }
                RawContent::Resource(resource_content) => {
                    if let Ok(val) = serde_json::to_value(&resource_content.resource) {
                        content_values.push(serde_json::json!({
                            "type": "resource",
                            "content": val,
                            "annotations": content.annotations
                        }));
                    }
                }
                _ => {}
            }
        }

        if content_values.is_empty() {
            Ok(serde_json::json!({ "status": "success" }))
        } else if content_values.len() == 1 {
            let val = content_values.remove(0);
            if val.is_object() {
                Ok(val)
            } else {
                Ok(serde_json::json!({ "content": val }))
            }
        } else {
            Ok(serde_json::json!({ "content": content_values }))
        }
    }

    async fn list_prompts(&self) -> Result<Vec<Prompt>, String> {
        let result = self.deref()
            .list_prompts(None)
            .await
            .map_err(|e| e.to_string())?;
        Ok(result.prompts)
    }

    async fn get_prompt(&self, name: String, args: Option<serde_json::Map<String, serde_json::Value>>) -> Result<GetPromptResult, String> {
        let params = GetPromptRequestParam {
            name: name.into(),
            arguments: args,
        };
        self.deref()
            .get_prompt(params)
            .await
            .map_err(|e| e.to_string())
    }

    async fn list_resources(&self) -> Result<Vec<Resource>, String> {
        let result = self.deref()
            .list_resources(None)
            .await
            .map_err(|e| e.to_string())?;
        Ok(result.resources)
    }

    async fn read_resource(&self, uri: String) -> Result<ReadResourceResult, String> {
        let params = ReadResourceRequestParam {
            uri: uri.into(),
        };
        self.deref()
            .read_resource(params)
            .await
            .map_err(|e| e.to_string())
    }
}

/// A helper to combine multiple MCP servers into one.
pub struct MultiMcpServer {
    servers: Vec<Box<dyn McpServer>>,
}

impl MultiMcpServer {
    pub fn new() -> Self {
        Self {
            servers: Vec::new(),
        }
    }

    pub fn add_server<S: McpServer + 'static>(mut self, server: S) -> Self {
        self.servers.push(Box::new(server));
        self
    }
    
    pub fn add_boxed_server(mut self, server: Box<dyn McpServer>) -> Self {
        self.servers.push(server);
        self
    }
}

#[async_trait]
impl McpServer for MultiMcpServer {
    async fn list_tools(&self) -> Result<Vec<Tool>, String> {
        let mut all_tools = Vec::new();
        for server in &self.servers {
            let tools = server.list_tools().await?;
            all_tools.extend(tools);
        }
        Ok(all_tools)
    }

    async fn call_tool(&self, name: String, args: Value) -> Result<Value, String> {
        for server in &self.servers {
            let tools = server.list_tools().await?;
            if tools.iter().any(|t| t.name == name) {
                return server.call_tool(name, args).await;
            }
        }
        Err(format!("Tool {} not found in any server", name))
    }

    async fn list_prompts(&self) -> Result<Vec<Prompt>, String> {
        let mut all_prompts = Vec::new();
        for server in &self.servers {
            let prompts = server.list_prompts().await?;
            all_prompts.extend(prompts);
        }
        Ok(all_prompts)
    }

    async fn get_prompt(&self, name: String, args: Option<serde_json::Map<String, Value>>) -> Result<GetPromptResult, String> {
        for server in &self.servers {
            let prompts = server.list_prompts().await?;
            if prompts.iter().any(|p| p.name == name) {
                return server.get_prompt(name, args).await;
            }
        }
        Err(format!("Prompt {} not found", name))
    }

    async fn list_resources(&self) -> Result<Vec<Resource>, String> {
        let mut all_resources = Vec::new();
        for server in &self.servers {
            let resources = server.list_resources().await?;
            all_resources.extend(resources);
        }
        Ok(all_resources)
    }

    async fn read_resource(&self, uri: String) -> Result<ReadResourceResult, String> {
        for server in &self.servers {
            let resources = server.list_resources().await?;
            if resources.iter().any(|r| r.uri == uri) {
                return server.read_resource(uri).await;
            }
        }
        Err(format!("Resource {} not found", uri))
    }
}
