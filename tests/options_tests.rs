use std::time::Duration;
use unia::options::{ModelOptions, TransportOptions};
use unia::providers::OpenAIModel;

#[test]
fn test_transport_options_builder() {
    let options = TransportOptions::new()
        .with_timeout(Duration::from_secs(30))
        .with_proxy("http://proxy.example.com".to_string())
        .with_header("X-Custom-Header".to_string(), "Value".to_string());

    match options {
        TransportOptions::Http {
            timeout,
            proxy,
            headers,
        } => {
            assert_eq!(timeout, Some(Duration::from_secs(30)));
            assert_eq!(proxy, Some("http://proxy.example.com".to_string()));

            let headers = headers.unwrap();
            assert_eq!(headers.get("X-Custom-Header"), Some(&"Value".to_string()));
        }
    }
}

#[test]
fn test_model_options_new() {
    let options: ModelOptions<OpenAIModel> = ModelOptions::new("gpt-5");

    assert_eq!(options.model, "gpt-5");
    assert_eq!(options.temperature, None);
    assert_eq!(options.max_tokens, None);
}

#[test]
fn test_model_options_custom() {
    let mut options = ModelOptions::<OpenAIModel>::new("gpt-5");
    options.temperature = Some(0.7);
    options.max_tokens = Some(100);

    assert_eq!(options.model, "gpt-5");
    assert_eq!(options.temperature, Some(0.7));
    assert_eq!(options.max_tokens, Some(100));
}
