use unai::{options::{ModelOptions, TransportOptions}, providers::{Gemini, Provider}};

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();
    let api_key = std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set");

    let model_options = ModelOptions {
        model: Some("gemini-2.5-flash".to_string()),
        system: Some("You are an expert English linguist, writer, and weather analyst. \
            You have access to tools to get real-time data. \
            Your goal is to provide high-quality, long-form literary analyses. \
            Always use your tools to get facts, then apply your persona to the output.".to_string()),
        ..Default::default()
    };

    let client = Gemini::create_with_options(api_key, model_options, TransportOptions::default());
    let uri = client.upload_file("text/plain", "Test file content".as_bytes()).await.unwrap();
    println!("Uploaded file URI: {}", uri);
}