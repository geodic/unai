use unai::model::{Message, Role, Part};
use unai::options::{ModelOptions, TransportOptions};
use unai::providers::{OpenAi, Provider};
use unai::{Agent};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");

    let model_options = ModelOptions {
        model: Some("gpt-4o".to_string()),
        ..Default::default()
    };

    // Create client with options
    let client = OpenAi::create_with_options(
        api_key,
        model_options,
        TransportOptions::default(),
    );

    // Create agent with client
    let agent = Agent::new(client);

    let messages = vec![Message::User(vec![Part::Text("Hello, world!".to_string())])];

    let response = agent.chat(messages, vec![]).await?;
    println!("Response: {:?}", response.data.first().and_then(|m| m.content()));

    Ok(())
}
