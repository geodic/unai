use async_trait::async_trait;
use rmcp::model::Tool;
use std::sync::{Arc, Mutex};
use unai::agent::Agent;
use unai::client::{Client, ClientError};
use unai::model::{FinishReason, Message, Part, Response, Usage};
use unai::options::{ModelOptions, TransportOptions};

#[derive(Clone)]
struct MockClient {
    responses: Arc<Mutex<Vec<Response>>>,
    requests: Arc<Mutex<Vec<Vec<Message>>>>,
}

impl MockClient {
    fn new(responses: Vec<Response>) -> Self {
        Self {
            responses: Arc::new(Mutex::new(responses)),
            requests: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

#[async_trait]
impl Client for MockClient {
    type ModelProvider = ();

    async fn request(
        &self,
        messages: Vec<Message>,
        _tools: Vec<Tool>,
    ) -> Result<Response, ClientError> {
        self.requests.lock().unwrap().push(messages);
        let mut responses = self.responses.lock().unwrap();
        if !responses.is_empty() {
            Ok(responses.remove(0))
        } else {
            Err(ClientError::ProviderError(
                "No more mock responses".to_string(),
            ))
        }
    }

    fn model_options(&self) -> &ModelOptions<Self::ModelProvider> {
        unimplemented!()
    }

    fn transport_options(&self) -> &TransportOptions {
        unimplemented!()
    }
}

#[tokio::test]
async fn test_agent_simple_chat() {
    let expected_response = Response {
        data: vec![Message::Assistant(vec![Part::Text {
            content: "Hello".to_string(),
            finished: true,
        }])],
        usage: Usage::default(),
        finish: FinishReason::Stop,
    };

    let client = MockClient::new(vec![expected_response]);
    let agent = Agent::new(client);

    let messages = vec![Message::User(vec![Part::Text {
        content: "Hi".to_string(),
        finished: true,
    }])];

    let response = agent.chat(messages).await.unwrap();

    assert_eq!(response.data.len(), 1);
    if let Message::Assistant(parts) = &response.data[0] {
        if let Part::Text { content, .. } = &parts[0] {
            assert_eq!(content, "Hello");
        } else {
            panic!("Expected text part");
        }
    } else {
        panic!("Expected assistant message");
    }
}
