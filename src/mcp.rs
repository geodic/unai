use crate::model::{MediaType, Message, Part};
use async_trait::async_trait;
use rmcp::model::{
    AnnotateAble, Annotated, CallToolRequestParam, GetPromptRequestParam, GetPromptResult, Prompt,
    PromptMessage, PromptMessageContent, PromptMessageRole, RawContent, ReadResourceRequestParam,
    ReadResourceResult, Resource, ResourceContents, Tool,
};
use rmcp::service::{RoleClient, RunningService};
use rmcp::ClientHandler;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::ops::Deref;
use thiserror::Error;
use uuid::Uuid;

#[derive(Debug, Error)]
pub enum MCPError {
    #[error("MCP error: {0}")]
    Mcp(String),
    #[error("Tool not found: {0}")]
    ToolNotFound(String),
    #[error("Prompt not found: {0}")]
    PromptNotFound(String),
    #[error("Resource not found: {0}")]
    ResourceNotFound(String),
    #[error("Server not found: {0}")]
    ServerNotFound(String),
    #[error("Server ID mismatch")]
    ServerIdMismatch,
}

/// A wrapper type that associates a value with an optional server ID.
///
/// This is used to track which MCP server a particular object (Tool, Prompt, Resource)
/// belongs to, especially when using a `MultiMCPServer`.
#[derive(Debug, Clone)]
pub struct Served<T> {
    /// The underlying value (Tool, Prompt, Resource, etc.).
    pub value: T,
    /// The ID of the server that provided this value.
    pub server_id: Option<String>,
}

impl<T> Served<T> {
    pub fn new(value: T, server_id: Option<String>) -> Self {
        Self { value, server_id }
    }
}

pub trait Servable {
    fn served(self, id: Option<String>) -> Served<Self>
    where
        Self: Sized,
    {
        Served::new(self, id)
    }
}

impl<T: AnnotateAble> Servable for Annotated<T> {}
impl Servable for Tool {}
impl Servable for Prompt {}
impl Servable for GetPromptResult {}
impl Servable for ReadResourceResult {}

/// Trait for MCP servers that can be used by the Agent.
#[async_trait]
pub trait MCPServer: Send + Sync {
    /// List available tools.
    async fn list_tools(&self) -> Result<Vec<Served<Tool>>, MCPError>;

    /// Execute a tool.
    async fn call_tool(
        &self,
        name: String,
        args: Value,
        server_id: Option<String>,
    ) -> Result<Part, MCPError>;

    /// List available prompts.
    async fn list_prompts(&self) -> Result<Vec<Served<Prompt>>, MCPError>;

    /// Get a prompt.
    async fn get_prompt(
        &self,
        prompt: &Served<Prompt>,
        args: Option<serde_json::Map<String, Value>>,
    ) -> Result<Served<GetPromptResult>, MCPError>;

    /// List available resources.
    async fn list_resources(&self) -> Result<Vec<Served<Resource>>, MCPError>;

    /// Read a resource.
    async fn read_resource(
        &self,
        resource: &Served<Resource>,
    ) -> Result<Served<ReadResourceResult>, MCPError>;
}

#[async_trait]
impl<S: ClientHandler + Send + Sync> MCPServer for RunningService<RoleClient, S> {
    async fn list_tools(&self) -> Result<Vec<Served<Tool>>, MCPError> {
        let result = self
            .deref()
            .list_tools(None)
            .await
            .map_err(|e| MCPError::Mcp(e.to_string()))?;
        Ok(result.tools.into_iter().map(|t| t.served(None)).collect())
    }

    async fn call_tool(
        &self,
        name: String,
        args: Value,
        _server_id: Option<String>,
    ) -> Result<Part, MCPError> {
        let params = CallToolRequestParam {
            name: name.clone().into(),
            arguments: args.as_object().cloned(),
        };

        let result = self
            .deref()
            .call_tool(params)
            .await
            .map_err(|e| MCPError::Mcp(e.to_string()))?;

        let mut structured = json!({});
        let mut parts = Vec::new();
        let mut parsed_text_content: Option<Value> = None;
        let mut raw_text_content: Vec<String> = Vec::new();

        for content in result.content {
            match content.raw {
                RawContent::Text(text_content) => {
                    if let Ok(parsed) = serde_json::from_str::<Value>(&text_content.text) {
                        parsed_text_content = Some(parsed);
                    } else {
                        raw_text_content.push(text_content.text);
                    }
                }
                RawContent::Image(image_content) => {
                    parts.push(Part::Media {
                        media_type: MediaType::Image,
                        data: image_content.data,
                        mime_type: image_content.mime_type,
                        uri: None,
                        finished: true,
                    });
                }
                RawContent::Resource(resource) => {
                    parts.push(Part::from(resource.resource));
                }
                _ => {}
            }
        }

        if let Some(s) = result.structured_content {
            structured = s;
        } else if let Some(parsed) = parsed_text_content {
            structured = parsed;
        } else if !raw_text_content.is_empty() {
            structured = json!({ "response": raw_text_content });
        }

        Ok(Part::FunctionResponse {
            id: None,
            name,
            response: structured,
            parts,
            finished: true,
        })
    }

    async fn list_prompts(&self) -> Result<Vec<Served<Prompt>>, MCPError> {
        let result = self
            .deref()
            .list_prompts(None)
            .await
            .map_err(|e| MCPError::Mcp(e.to_string()))?;
        Ok(result.prompts.into_iter().map(|p| p.served(None)).collect())
    }

    async fn get_prompt(
        &self,
        prompt: &Served<Prompt>,
        args: Option<serde_json::Map<String, Value>>,
    ) -> Result<Served<GetPromptResult>, MCPError> {
        let params = GetPromptRequestParam {
            name: prompt.value.name.clone(),
            arguments: args,
        };
        self.deref()
            .get_prompt(params)
            .await
            .map(|r| r.served(None))
            .map_err(|e| MCPError::Mcp(e.to_string()))
    }

    async fn list_resources(&self) -> Result<Vec<Served<Resource>>, MCPError> {
        let result = self
            .deref()
            .list_resources(None)
            .await
            .map_err(|e| MCPError::Mcp(e.to_string()))?;
        Ok(result
            .resources
            .into_iter()
            .map(|r| r.served(None))
            .collect())
    }

    async fn read_resource(
        &self,
        resource: &Served<Resource>,
    ) -> Result<Served<ReadResourceResult>, MCPError> {
        let params = ReadResourceRequestParam {
            uri: resource.value.uri.clone(),
        };
        self.deref()
            .read_resource(params)
            .await
            .map(|r| r.served(None))
            .map_err(|e| MCPError::Mcp(e.to_string()))
    }
}

/// A helper to combine multiple MCP servers into one.
pub struct MultiMCPServer {
    servers: HashMap<String, Box<dyn MCPServer>>,
}

impl Default for MultiMCPServer {
    fn default() -> Self {
        Self::new()
    }
}

impl MultiMCPServer {
    pub fn new() -> Self {
        Self {
            servers: HashMap::new(),
        }
    }

    pub fn from_servers(servers: Vec<Box<dyn MCPServer>>) -> Self {
        let mut map = HashMap::new();
        for server in servers {
            let id = Uuid::new_v4().to_string();
            map.insert(id, server);
        }
        Self { servers: map }
    }

    pub fn add_server<S: MCPServer + 'static>(mut self, server: S) -> Self {
        let id = Uuid::new_v4().to_string();
        self.servers.insert(id, Box::new(server));
        self
    }

    pub fn add_boxed_server(mut self, server: Box<dyn MCPServer>) -> Self {
        let id = Uuid::new_v4().to_string();
        self.servers.insert(id, server);
        self
    }
}

#[async_trait]
impl MCPServer for MultiMCPServer {
    async fn list_tools(&self) -> Result<Vec<Served<Tool>>, MCPError> {
        let mut all_tools = Vec::new();
        for (id, server) in &self.servers {
            let tools: Vec<Served<Tool>> = server.list_tools().await?;
            all_tools.extend(tools.into_iter().map(|mut t| {
                t.server_id = Some(id.clone());
                t
            }));
        }
        Ok(all_tools)
    }

    async fn call_tool(
        &self,
        name: String,
        args: Value,
        server_id: Option<String>,
    ) -> Result<Part, MCPError> {
        if let Some(id) = server_id {
            if let Some(server) = self.servers.get(&id) {
                return server.call_tool(name, args, None).await;
            }
            return Err(MCPError::ServerNotFound(id));
        }

        for server in self.servers.values() {
            let tools: Vec<Served<Tool>> = server.list_tools().await?;
            if tools.iter().any(|t| t.value.name == name) {
                return server.call_tool(name, args, None).await;
            }
        }
        Err(MCPError::ToolNotFound(name))
    }

    async fn list_prompts(&self) -> Result<Vec<Served<Prompt>>, MCPError> {
        let mut all_prompts = Vec::new();
        for (id, server) in &self.servers {
            let prompts: Vec<Served<Prompt>> = server.list_prompts().await?;
            all_prompts.extend(prompts.into_iter().map(|mut p| {
                p.server_id = Some(id.clone());
                p
            }));
        }
        Ok(all_prompts)
    }

    async fn get_prompt(
        &self,
        prompt: &Served<Prompt>,
        args: Option<serde_json::Map<String, Value>>,
    ) -> Result<Served<GetPromptResult>, MCPError> {
        if let Some(id) = &prompt.server_id {
            if let Some(server) = self.servers.get(id) {
                return server.get_prompt(prompt, args).await;
            }
            return Err(MCPError::ServerNotFound(id.clone()));
        }
        Err(MCPError::ServerIdMismatch)
    }

    async fn list_resources(&self) -> Result<Vec<Served<Resource>>, MCPError> {
        let mut all_resources = Vec::new();
        for (id, server) in &self.servers {
            let resources: Vec<Served<Resource>> = server.list_resources().await?;
            all_resources.extend(resources.into_iter().map(|mut r| {
                r.server_id = Some(id.clone());
                r
            }));
        }
        Ok(all_resources)
    }

    async fn read_resource(
        &self,
        resource: &Served<Resource>,
    ) -> Result<Served<ReadResourceResult>, MCPError> {
        if let Some(id) = &resource.server_id {
            if let Some(server) = self.servers.get(id) {
                return server.read_resource(resource).await;
            }
            return Err(MCPError::ServerNotFound(id.clone()));
        }
        Err(MCPError::ServerIdMismatch)
    }
}

#[async_trait]
pub trait AttachResources {
    async fn resources(
        self,
        server: &dyn MCPServer,
        resources: Vec<Served<Resource>>,
    ) -> Result<Self, MCPError>
    where
        Self: Sized;
}

#[async_trait]
impl AttachResources for Message {
    async fn resources(
        mut self,
        server: &dyn MCPServer,
        resources: Vec<Served<Resource>>,
    ) -> Result<Self, MCPError> {
        for resource in resources {
            let result = server.read_resource(&resource).await?;
            let parts: Vec<Part> = result.into();
            self.parts_mut().extend(parts);
        }
        Ok(self)
    }
}

#[async_trait]
impl AttachResources for Vec<Message> {
    async fn resources(
        mut self,
        server: &dyn MCPServer,
        resources: Vec<Served<Resource>>,
    ) -> Result<Self, MCPError> {
        if !self.is_empty() {
            let first = self.remove(0);
            let first = first.resources(server, resources).await?;
            self.insert(0, first);
        }
        Ok(self)
    }
}

impl From<ResourceContents> for Part {
    fn from(resource: ResourceContents) -> Self {
        match resource {
            ResourceContents::TextResourceContents {
                text,
                mime_type,
                uri,
                ..
            } => Part::Media {
                media_type: MediaType::Text,
                data: text,
                mime_type: mime_type.unwrap_or_else(|| "text/plain".to_string()),
                uri: Some(uri),
                finished: true,
            },
            ResourceContents::BlobResourceContents {
                blob,
                mime_type,
                uri,
                ..
            } => {
                let mime = mime_type.unwrap_or_else(|| "application/octet-stream".to_string());
                let media_type = if mime.starts_with("image/") {
                    MediaType::Image
                } else if mime == "application/pdf" {
                    MediaType::Document
                } else {
                    MediaType::Binary
                };

                Part::Media {
                    media_type,
                    data: blob,
                    mime_type: mime,
                    uri: Some(uri),
                    finished: true,
                }
            }
        }
    }
}

impl From<PromptMessage> for Message {
    fn from(pm: PromptMessage) -> Self {
        let part = match pm.content {
            PromptMessageContent::Text { text } => Part::Text {
                content: text,
                finished: true,
            },
            PromptMessageContent::Image { image, .. } => Part::Media {
                media_type: MediaType::Image,
                data: image.data.clone(),
                mime_type: image.mime_type.clone(),
                uri: None,
                finished: true,
            },
            PromptMessageContent::Resource { resource } => Part::from(resource.resource.clone()),
            PromptMessageContent::ResourceLink { .. } => {
                unimplemented!("ResourceLink not supported")
            }
        };

        match pm.role {
            PromptMessageRole::User => Message::User(vec![part]),
            PromptMessageRole::Assistant => Message::Assistant(vec![part]),
        }
    }
}

impl From<Served<GetPromptResult>> for Vec<Message> {
    fn from(served: Served<GetPromptResult>) -> Self {
        served
            .value
            .messages
            .into_iter()
            .map(Message::from)
            .collect()
    }
}

impl From<Served<ReadResourceResult>> for Vec<Part> {
    fn from(served: Served<ReadResourceResult>) -> Self {
        served.value.contents.into_iter().map(Part::from).collect()
    }
}
