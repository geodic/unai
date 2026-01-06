use rmcp::{transport::StreamableHttpClientTransport, ServiceExt};
use unai::{
    mcp::MCPServer,
    providers::{openai::OpenAI, Provider},
    Agent,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ============================================================================================
    // Step 1: Setup Client
    // ============================================================================================
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");
    let client = OpenAI::create(api_key, "gpt-5".to_string());
    let agent = Agent::new(client);

    // ============================================================================================
    // Step 2: Connect to External MCP Server (Streamable HTTP)
    // ============================================================================================
    // We connect to the hosted GitHub MCP server via Streamable HTTP.
    // URL: https://api.githubcopilot.com/mcp/
    //
    // For more information on how to connect to different MCP servers (stdio, SSE, etc.),
    // please refer to the `rmcp` documentation.

    println!("Connecting to GitHub MCP server...");

    let transport = StreamableHttpClientTransport::from_uri("https://api.githubcopilot.com/mcp/");

    let mcp_client = ().serve(transport).await?;

    // Use the rmcp client directly as it implements MCPServer
    let mcp_server = mcp_client;

    // ============================================================================================
    // Step 3: List Prompts
    // ============================================================================================
    // Prompts are reusable templates provided by the server.
    println!("Listing prompts...");
    // We use the MCPServer trait method here
    let prompts = mcp_server.list_prompts().await?;
    for prompt in &prompts {
        println!("- {}", prompt.value.name);
    }

    // ============================================================================================
    // Step 4: List Resources
    // ============================================================================================
    // Resources are data sources (files, logs, etc.) exposed by the server.
    println!("Listing resources...");
    let resources = mcp_server.list_resources().await?;
    for resource in &resources {
        println!("- {}", resource.value.uri);
    }

    // ============================================================================================
    // Step 5: Use Prompts and Resources with Agent
    // ============================================================================================

    // Example: Get a prompt and convert it to messages
    if let Some(first_prompt) = prompts.first() {
        println!("Getting prompt: {}", first_prompt.value.name);
        let prompt_result = mcp_server.get_prompt(first_prompt, None).await?;

        // Convert Served<GetPromptResult> to Vec<Message> using From implementation
        let messages: Vec<unai::model::Message> = prompt_result.into();
        println!("Converted prompt to {} messages", messages.len());

        // You can now pass these messages to the agent
        // let response = agent.chat("...", messages).await?;
    }

    // Example: Read a resource and convert it to parts
    if let Some(first_resource) = resources.first() {
        println!("Reading resource: {}", first_resource.value.uri);
        let resource_result = mcp_server.read_resource(first_resource).await?;

        // Convert Served<ReadResourceResult> to Vec<Part> using From implementation
        let parts: Vec<unai::model::Part> = resource_result.into();
        println!("Converted resource to {} parts", parts.len());

        // You can attach these parts to a user message
        // let message = unai::model::Message::User(parts);
    }

    let _agent = agent.with_server(mcp_server);

    // The agent can now use tools provided by GitHub.
    // You can use `agent.chat(...)` to interact with the tools.

    Ok(())
}
