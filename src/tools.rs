//! Tool system for automatic function calling with typed input/output.

use async_trait::async_trait;
pub use rmcp::model::Tool;
use serde_json::Value;

/// Error type for tool execution.
#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    #[error("Tool error: {0}")]
    Error(String),
}

/// Trait for tools that can be called by LLMs.
#[async_trait]
pub trait ToolService: Send + Sync {
    /// List available tools.
    async fn list_tools(&self) -> Result<Vec<Tool>, ToolError>;

    /// Execute a tool.
    async fn call_tool(&self, name: String, args: Value) -> Result<Value, ToolError>;
}
