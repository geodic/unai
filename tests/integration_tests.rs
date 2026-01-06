use unai::client::Client;
use unai::model::{Message, Part, Role};
use unai::providers::{OpenAI, Provider};

#[test]
fn test_client_creation() {
    let client = OpenAI::create("test-key".to_string(), "gpt-5".to_string());
    assert_eq!(client.model_options().model, "gpt-5");
}

#[test]
fn test_message_construction() {
    let msg = Message::User(vec![Part::Text {
        content: "Hello".to_string(),
        finished: true,
    }]);

    assert_eq!(msg.role(), Role::User);
    if let Message::User(parts) = msg {
        assert_eq!(parts.len(), 1);
        if let Part::Text { content, .. } = &parts[0] {
            assert_eq!(content, "Hello");
        } else {
            panic!("Expected Text part");
        }
    } else {
        panic!("Expected User message");
    }
}
