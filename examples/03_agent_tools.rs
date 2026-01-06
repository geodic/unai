use rmcp::service::ServiceExt;
use rmcp::{
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{ServerCapabilities, ServerInfo},
    schemars, tool, tool_handler, tool_router, ServerHandler,
};
use serde::Deserialize;
use unai::model::{Message, Part};
use unai::providers::{OpenAI, Provider};
use unai::Agent;

// ============================================================================================
// Step 1: Define Tool Arguments
// ============================================================================================
// We use `serde` and `schemars` to define the structure of the arguments our tool expects.
// This schema will be automatically converted to JSON Schema and sent to the LLM.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct WeatherArgs {
    #[schemars(description = "The location to get the weather for")]
    pub location: String,
    #[schemars(description = "The unit (celsius or fahrenheit)")]
    pub unit: String,
}

// ============================================================================================
// Step 2: Define the Tool Handler
// ============================================================================================
// We create a struct to hold our tool logic. The `ToolRouter` handles dispatching calls.
//
// Note: This is the standard way to define an MCP server using the `rmcp` library.
// For more details on how `rmcp` servers work, please refer to the `rmcp` documentation.
#[derive(Debug, Clone)]
pub struct WeatherTools {
    tool_router: ToolRouter<Self>,
}

#[tool_router]
impl WeatherTools {
    pub fn new() -> Self {
        Self {
            tool_router: Self::tool_router(),
        }
    }

    // The `#[tool]` attribute marks this method as an exposed tool.
    // The description is provided to the LLM to help it understand when to use the tool.
    #[tool(description = "Get the current weather for a location")]
    fn get_weather(
        &self,
        Parameters(WeatherArgs { location, unit }): Parameters<WeatherArgs>,
    ) -> String {
        println!("> Tool called: get_weather({}, {})", location, unit);

        // In a real application, you would call an external weather API here.
        let response = serde_json::json!({
            "temperature": 22,
            "condition": "Sunny",
            "location": location,
            "unit": unit
        });
        response.to_string()
    }
}

// We implement `ServerHandler` to provide server metadata and capabilities.
#[tool_handler]
impl ServerHandler for WeatherTools {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            server_info: rmcp::model::Implementation {
                name: "weather-server".into(),
                version: "1.0".into(),
                ..Default::default()
            },
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            ..Default::default()
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ============================================================================================
    // Step 3: Setup Client and Agent
    // ============================================================================================
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");
    let client = OpenAI::create(api_key, "gpt-5".to_string());
    let agent = Agent::new(client);

    // ============================================================================================
    // Step 4: Setup MCP Server
    // ============================================================================================
    // We create an in-memory MCP server using `rmcp`.
    // We use a `tokio::io::duplex` channel to connect the client (Agent) and the server
    // without needing a network connection.
    let tools_handler = WeatherTools::new();
    let (client_transport, server_transport) = tokio::io::duplex(1024);

    // Spawn the server in a background task
    tokio::spawn(async move {
        let service = tools_handler
            .serve(server_transport)
            .await
            .expect("Failed to start server");
        service.waiting().await.expect("Server error");
    });

    // Create the client side of the MCP connection
    let tools = ().serve(client_transport).await?;

    // ============================================================================================
    // Step 5: Attach Server to Agent
    // ============================================================================================
    // We attach the MCP client to the Agent.
    // The Agent will now automatically discover tools from this server.
    let agent = agent.with_server(tools);

    // ============================================================================================
    // Step 6: Run the Agent
    // ============================================================================================
    // The agent will:
    // 1. Send the user message to the LLM.
    // 2. Receive a tool call request from the LLM.
    // 3. Execute the tool on our local server.
    // 4. Send the tool result back to the LLM.
    // 5. Return the final response.
    let response = agent
        .chat(vec![Message::User(vec![Part::Text {
            content: "What is the weather in Tokyo in celsius?".to_string(),
            finished: true,
        }])])
        .await?;

    let content = response
        .data
        .last()
        .and_then(|m| m.content())
        .unwrap_or_default();
    println!("Agent: {}", content);

    Ok(())
}
