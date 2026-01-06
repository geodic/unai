use unai::{
    model::{Message, Part},
    providers::{openai::OpenAI, Provider},
    Client,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ============================================================================================
    // Step 1: Setup the Provider
    // ============================================================================================
    // Unai is designed to be provider-agnostic. You can switch between OpenAI, Anthropic, Gemini,
    // and others by simply changing the provider struct (e.g., `OpenAI` -> `Anthropic`).
    //
    // The `Provider` trait defines a `create` method that takes your API key and the model name.
    // This returns a client that implements the `Client` trait.
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");
    let client = OpenAI::create(api_key, "gpt-5".to_string());

    // ============================================================================================
    // Step 2: Construct the Message
    // ============================================================================================
    // Messages in Unai are structured to support multimodal content (text, images, files).
    // A `Message` is an enum representing the role (User, Assistant, System, Tool).
    // Each message contains a vector of `Part`s.
    //
    // Here, we create a simple User message with a single Text part.
    let messages = vec![Message::User(vec![Part::Text {
        content: "Explain quantum computing in one sentence.".to_string(),
        finished: true, // `finished` indicates if the part is complete (relevant for streaming)
    }])];

    println!("Sending request...");

    // ============================================================================================
    // Step 3: Send the Request
    // ============================================================================================
    // The `request` method is part of the `Client` trait. It sends the conversation history
    // to the LLM and awaits the response.
    //
    // The second argument is for `tools`, which we are leaving empty for this basic example.
    let response = client.request(messages, vec![]).await?;

    // ============================================================================================
    // Step 4: Process the Response
    // ============================================================================================
    // The `Response` object contains:
    // - `data`: A list of generated messages (usually just one Assistant message).
    // - `usage`: Token usage statistics (prompt tokens, completion tokens).
    // - `finish`: The reason generation stopped (e.g., "stop", "length").
    //
    // We extract the text content from the first message in the response.
    let content = response
        .data
        .first()
        .and_then(|m| m.content())
        .unwrap_or_default();
    println!("Response: {}", content);

    Ok(())
}
