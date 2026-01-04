use unai::model::{Message, Part, Role};
use unai::options::{ModelOptions, TransportOptions};
use unai::providers::{Anthropic, Provider};
use unai::Agent;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY must be set");

    let model_options = ModelOptions {
        model: Some("claude-3-opus-20240229".to_string()),
        ..Default::default()
    };

    let client =
        Anthropic::create_with_options(api_key, model_options, TransportOptions::default());
    let agent = Agent::new(client);

    let messages = vec![Message::User(vec![Part::Text { content: "Hello, world!".to_string(), finished: true }])];

    let response = agent.chat(messages).await?;
    println!(
        "Response: {:?}",
        response.data.first().and_then(|m| m.content())
    );

    Ok(())
}
