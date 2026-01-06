use base64::{engine::general_purpose, Engine as _};
use unia::{
    model::{MediaType, Message, Part},
    providers::{openai::OpenAI, Provider},
    Client,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ============================================================================================
    // Step 1: Setup Provider
    // ============================================================================================
    // Multimodal capabilities (understanding images, audio, etc.) depend on the model.
    // Ensure you select a model that supports vision, such as GPT-5, Claude 4.5 Haiku, or Gemini 3 Flash.
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");
    let client = OpenAI::create(api_key, "gpt-5".to_string());

    // ============================================================================================
    // Step 2: Fetch Image Data
    // ============================================================================================
    // Most LLM APIs require images to be sent as base64-encoded strings or via a URL.
    // unia handles the formatting, but you need to provide the raw data or the URI.
    //
    // Here, we download an image from the web and encode it as base64.
    let image_url = "https://picsum.photos/id/13/2500/1667";
    println!("Fetching image from {}...", image_url);

    let image_bytes = reqwest::get(image_url).await?.bytes().await?;
    let base64_image = general_purpose::STANDARD.encode(&image_bytes);

    // ============================================================================================
    // Step 3: Create a Multimodal Message
    // ============================================================================================
    // A `Message` can contain multiple `Part`s. To send an image, we use `Part::Media`.
    //
    // - `media_type`: The type of media (Image, Audio, Video).
    // - `data`: The base64 encoded data string.
    // - `mime_type`: The MIME type of the file (e.g., "image/jpeg", "image/png").
    // - `uri`: Optional URI to tell the model where the media is located or came from.
    let message = Message::User(vec![
        Part::Text {
            content: "What is in this image?".to_string(),
            finished: true,
        },
        Part::Media {
            media_type: MediaType::Image,
            data: base64_image,
            mime_type: "image/jpeg".to_string(),
            uri: Some(image_url.to_string()), // We provide the URI for context
            finished: true,
        },
    ]);

    println!("Analyzing image...");

    // ============================================================================================
    // Step 4: Send Request
    // ============================================================================================
    // The client will automatically format the request for the specific provider.
    // For OpenAI, it converts the `Part::Media` into the correct `image_url` object.
    let response = client.request(vec![message], vec![]).await?;

    let content = response
        .data
        .first()
        .and_then(|m| m.content())
        .unwrap_or_default();
    println!("Response: {}", content);

    Ok(())
}
