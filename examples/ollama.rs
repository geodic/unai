use unai::model::{Message, Part, Role};
use unai::options::{ModelOptions, TransportOptions};
use unai::providers::{Ollama, Provider};
use unai::Agent;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Ollama usually runs locally, so API key might not be needed, but we'll check for base URL
    let base_url =
        std::env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| "http://localhost:11434".to_string());

    let model_options = ModelOptions {
        model: Some("llama3".to_string()),
        ..Default::default()
    };

    let client = Ollama::create_with_options(base_url, model_options, TransportOptions::default());
    let agent = Agent::new(client);

    let messages = vec![Message::User(vec![Part::Text { content: "Hello, world!".to_string(), finished: true }])];

    let response = agent.chat(messages).await?;
    println!(
        "Response: {:?}",
        response.data.first().and_then(|m| m.content())
    );

    Ok(())
}
