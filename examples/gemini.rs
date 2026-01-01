use rmcp::{
    ServerHandler,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{ServerCapabilities, ServerInfo},
    schemars, tool, tool_handler, tool_router,
};
use rmcp::service::ServiceExt;
use unai::model::{Message, Role, Part};
use unai::options::{ModelOptions, TransportOptions};
use unai::providers::{Gemini, Provider};
use unai::{Agent};
use serde::Deserialize;

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct WeatherArgs {
    #[schemars(description = "The location to get the weather for")]
    pub location: String,
}

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

    #[tool(description = "Get the weather for a location")]
    fn get_weather(&self, Parameters(WeatherArgs { location }): Parameters<WeatherArgs>) -> String {
        let response = serde_json::json!({
            "condition": "sunny",
            "temperature": 72,
            "location": location
        });
        response.to_string()
    }
}

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
    tracing_subscriber::fmt::init();
    let api_key = std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set");

    let model_options = ModelOptions {
        model: Some("gemini-2.5-flash".to_string()),
        ..Default::default()
    };

    let client = Gemini::create_with_options(
        api_key,
        model_options,
        TransportOptions::default(),
    );
    let agent = Agent::new(client);

    let messages = vec![Message::User(vec![Part::Text("Can you give me an analysis on the weather in Tokyo? Please get the weather from the tool and analyse it.".to_string())])];

    let tools_handler = WeatherTools::new();
    
    let (client_transport, server_transport) = tokio::io::duplex(1024);

    tokio::spawn(async move {
        tracing::info!("Starting weather tool server...");
        let service = tools_handler.serve(server_transport).await.expect("Failed to start server");
        service.waiting().await.expect("Server error");
    });

    let tools = ().serve(client_transport).await?;

    let agent = agent.with_mcp_server(tools);

    let response = agent.chat(messages, vec![]).await?;
    
    for message in &response.data {
        for part in message.parts() {
            match part {
                Part::Text(content) => println!("Response: {}", content),
                Part::FunctionCall { name, arguments, .. } => println!("Tool Call: {} {}", name, arguments),
                Part::FunctionResponse { name, response, .. } => println!("Tool Result: {} {}", name, response),
                _ => {}
            }
        }
    }

    Ok(())
}
