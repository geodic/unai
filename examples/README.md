# unia Examples

This directory contains tutorial-like examples demonstrating the core features of the `unia` library.

## Running Examples

You can run any example using `cargo run --example <name>`.

Most examples require an API key to be set in your environment variables (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`).

## Available Examples

### 1. Basic Client (`01_basic_client.rs`)
Demonstrates the fundamental usage of the `Client` trait.
- Creating a Provider (OpenAI, Anthropic, Gemini, etc.)
- Creating a Client
- Sending a simple text message
- Handling the response
- Run: `cargo run --example 01_basic_client`

### 2. Streaming (`02_streaming.rs`)
Shows how to consume streaming responses using Server-Sent Events (SSE).
- Using `send_stream`
- Iterating over `StreamChunk`s
- Real-time output
- Run: `cargo run --example 02_streaming`

### 3. Agent & Tools (`03_agent_tools.rs`)
Demonstrates the high-level `Agent` abstraction and the `#[tool]` macro.
- Defining a tool function with `#[tool]`
- Creating an `Agent`
- Registering tools
- Automatic tool execution loop
- Run: `cargo run --example 03_agent_tools`

### 4. Multimodal (`04_multimodal.rs`)
Shows how to send non-text content like images.
- Constructing `Message` with multiple `Part`s
- Sending Image URLs (or base64 data)
- Vision capabilities
- Run: `cargo run --example 04_multimodal`

### 5. MCP Features (`05_mcp_features.rs`)
Demonstrates integration with the Model Context Protocol (MCP).
- Connecting to an external MCP server (e.g., GitHub Copilot)
- Listing available prompts and resources
- Using MCP tools with an Agent
- Run: `cargo run --example 05_mcp_features`
