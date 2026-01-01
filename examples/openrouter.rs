use unai::model::{Message, Role, Part};
use unai::options::{ModelOptions, TransportOptions};
use unai::providers::{OpenRouter, Provider};
use unai::{Agent};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("OPENROUTER_API_KEY").expect("OPENROUTER_API_KEY must be set");

    let model_options = ModelOptions {
        model: Some("openai/gpt-3.5-turbo".to_string()),
        ..Default::default()
    };

    let client = OpenRouter::create_with_options(api_key, model_options, TransportOptions::default());
    let agent = Agent::new(client);

    let messages = vec![Message::User(vec![Part::Text("Hello, world!".to_string())])];

    let response = agent.chat(messages, vec![]).await?;
    println!("Response: {:?}", response.data.first().and_then(|m| m.content()));

    Ok(())
}
