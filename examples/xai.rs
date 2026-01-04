use unai::model::{Message, Part, Role};
use unai::options::{ModelOptions, TransportOptions};
use unai::providers::{Provider, XAIModel, XAI};
use unai::Agent;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("XAI_API_KEY").expect("XAI_API_KEY must be set");

    let model_options = ModelOptions {
        model: Some("grok-beta".to_string()),
        provider: XAIModel::default(),
        ..Default::default()
    };

    let client = XAI::create_with_options(api_key, model_options, TransportOptions::default());
    let agent = Agent::new(client);

    let messages = vec![Message::User(vec![Part::Text { content: "Hello, world!".to_string(), finished: true }])];

    let response = agent.chat(messages).await?;
    println!(
        "Response: {:?}",
        response.data.first().and_then(|m| m.content())
    );

    Ok(())
}
