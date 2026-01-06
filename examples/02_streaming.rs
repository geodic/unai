use futures::StreamExt;
use std::io::{self, Write};
use unai::{
    model::{Message, Part},
    providers::{openai::OpenAI, Provider},
    StreamingClient,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ============================================================================================
    // Step 1: Setup Provider
    // ============================================================================================
    // Streaming allows you to receive the response in chunks as it is generated, which provides
    // a better user experience for long responses.
    //
    // The `StreamingClient` trait adds the `request_stream` method to the client.
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");
    let client = OpenAI::create(api_key, "gpt-5".to_string());

    let messages = vec![Message::User(vec![Part::Text {
        content: "Write a haiku about Rust programming.".to_string(),
        finished: true,
    }])];

    println!("Streaming response...");

    // ============================================================================================
    // Step 2: Initiate Stream
    // ============================================================================================
    // `request_stream` sends the request and returns a `Stream` (from the `futures` crate).
    // This stream yields `Result<Response, ClientError>` items.
    //
    // IMPORTANT: Unlike many other libraries that yield "deltas" (just the new characters),
    // Unai yields the **entire generated response object so far** in every iteration.
    // This makes it easier to reason about the state of the response, but requires you to
    // calculate the difference if you want to print only the new characters.
    let mut stream = client.request_stream(messages, vec![]).await?;

    // ============================================================================================
    // Step 3: Consume Stream
    // ============================================================================================
    // We use `while let Some(...)` to iterate over the stream until it is exhausted.
    let mut last_len = 0;

    while let Some(result) = stream.next().await {
        match result {
            Ok(response) => {
                // Extract the full text content generated so far.
                let content = response
                    .data
                    .first()
                    .and_then(|m| m.content())
                    .unwrap_or_default();

                // Calculate the new text added in this chunk.
                if content.len() > last_len {
                    let new_content = &content[last_len..];
                    print!("{}", new_content);
                    io::stdout().flush()?;
                    last_len = content.len();
                }
            }
            Err(e) => {
                eprintln!("\nError: {}", e);
                break;
            }
        }
    }
    println!(); // Add a final newline

    Ok(())
}
