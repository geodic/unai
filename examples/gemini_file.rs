use unai::model::{Message, Part};
use unai::options::{ModelOptions, TransportOptions};
use unai::providers::{Gemini, Provider};
use unai::Agent;
use base64::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing subscriber for logging
    tracing_subscriber::fmt::init();

    let api_key = std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set");

    let client = Gemini::create_with_options(
        api_key,
        ModelOptions {
            model: "gemini-2.5-flash".to_string().into(),
            ..Default::default()
        },
        TransportOptions::default(),
    );

    let agent = Agent::new(client);

    // Create a sample text file content
    let file_content = "This is a secret document. The password is 'banana'.";
    let base64_data = BASE64_STANDARD.encode(file_content);

    let messages = vec![Message::User(vec![
        Part::File {
            data: base64_data,
            mime_type: "text/plain".to_string(),
            uri: None, // The client should handle uploading this
            finished: true,
        },
        Part::Text {
            content: "What is the password mentioned in the file?".to_string(),
            finished: true,
        },
    ])];

    println!("Sending request with file...");
    let response = agent.chat(messages).await?;

    println!("Response:");
    if let Some(message) = response.data.first() {
        if let Some(content) = message.content() {
            println!("{}", content);
        } else {
            println!("No text content in response.");
        }
    } else {
        println!("No messages in response.");
    }

    Ok(())
}
