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
//! - [`Provider`](crate::providers::Provider): Factory trait for creating clients.
//! - [`Client`]: Trait for making requests to LLM providers.
//! - [`Agent`]: High-level orchestration for multi-turn conversations and tool use.
//! - [`ModelOptions`](crate::options::ModelOptions): Model behavior parameters (temperature, max_tokens, etc.)
//! - [`TransportOptions`](crate::options::TransportOptions): Transport configuration (timeout, proxy, etc.)
//! - [`Message`]: Individual conversation messages with role and content
//!
//! ## Example
//! ```no_run
//! use unai::client::Client;
//! use unai::model::{Message, Part};
//! use unai::providers::{OpenAI, Provider};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create client using the factory
//!     let client = OpenAI::create("your-api-key".to_string(), "gpt-5".to_string());
//!     
//!     // Create a message with text content
//!     let messages = vec![
//!         Message::User(vec![
//!             Part::Text {
//!                 content: "Hello!".to_string(),
//!                 finished: true,
//!             }
//!         ])
//!     ];
//!     
//!     // Send request
//!     let response = client.request(messages, vec![]).await?;
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
pub use mcp::{AttachResources, MCPServer};
pub use model::{GeneralRequest, Message, Response};
pub use tools::{Tool, ToolError, ToolService};

// Re-export rmcp for convenience
pub use rmcp;
