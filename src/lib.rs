//! # unai - Universal AI Client Library
//!
//! A small, pragmatic Rust library providing a provider-agnostic LLM client architecture
//! with a fully generic options system.
//!
//! ## Features
//! - Async-first, tokio compatible
//! - Provider-agnostic trait-based design
//! - Generic model and transport options
//! - Streaming support via Server-Sent Events
//! - Type-safe request/response models
//!
//! ## Architecture
//!
//! The library uses a factory-based design:
//!
//! 1. **Providers** act as factories to create Clients.
//! 2. **Clients** store authentication and configuration state.
//! 3. **Agents** wrap Clients to provide automatic tool execution loops.
//!
//! ### Core Types
//!
//! - **`Provider`**: Factory trait for creating clients.
//! - **`Client`**: Trait for making requests to LLM providers.
//! - **`ModelOptions<T>`**: Model behavior parameters (temperature, max_tokens, etc.)
//! - **`TransportOptions`**: Transport configuration (timeout, proxy, etc.)
//! - **Message**: Individual conversation messages with role and content
//!
//! ## Example
//! ```no_run
//! use unai::client::Client;
//! use unai::model::{Message, Role};
//! use unai::providers::{OpenAiProvider, Provider};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create client using the factory
//!     let client = OpenAiProvider::create("your-api-key".to_string());
//!     
//!     // Use convenient instance method with just messages
//!     let messages = vec![
//!         Message::Text {
//!             role: Role::User,
//!             content: "Hello!".to_string(),
//!         }
//!     ];
//!     
//!     // Use the chat helper which creates a Context internally
//!     let response = client.chat(messages).await?;
//!     println!("{:?}", response);
//!     Ok(())
//! }
//! ```

pub mod agent;
pub mod api;
pub mod client;
pub mod http;
pub mod mcp;
pub mod model;
pub mod options;
pub mod providers;
pub mod sse;
pub mod stream;
pub mod tools;

pub use agent::Agent;
pub use client::{Client, ClientError, StreamingClient};
pub use mcp::{MCPServer, AttachResources};
pub use model::{GeneralRequest, Message, Response};
// pub use stream::StreamChunk;
pub use tools::{Tool, ToolError, ToolService};

// Re-export rmcp for convenience
pub use rmcp;

// Re-export the proc macro attribute
pub use unai_macros::tool;
