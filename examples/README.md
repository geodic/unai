# unai Examples

This directory contains simple examples for each supported provider.

## Running Examples

To run an example, you need to set the corresponding API key environment variable.

### OpenAI
```bash
export OPENAI_API_KEY=sk-...
cargo run --example openai
```

### Anthropic
```bash
export ANTHROPIC_API_KEY=sk-ant-...
cargo run --example anthropic
```

### Google Gemini
```bash
export GEMINI_API_KEY=...
cargo run --example gemini
```

### Groq
```bash
export GROQ_API_KEY=gsk_...
cargo run --example groq
```

### Mistral
```bash
export MISTRAL_API_KEY=...
cargo run --example mistral
```

### DeepSeek
```bash
export DEEPSEEK_API_KEY=...
cargo run --example deepseek
```

### Perplexity
```bash
export PERPLEXITY_API_KEY=pplx-...
cargo run --example perplexity
```

### OpenRouter
```bash
export OPENROUTER_API_KEY=sk-or-...
cargo run --example openrouter
```

### Together AI
```bash
export TOGETHER_API_KEY=...
cargo run --example together
```

### Fireworks AI
```bash
export FIREWORKS_API_KEY=...
cargo run --example fireworks
```

### Hyperbolic
```bash
export HYPERBOLIC_API_KEY=...
cargo run --example hyperbolic
```

### Moonshot AI
```bash
export MOONSHOT_API_KEY=...
cargo run --example moonshot
```

### xAI (Grok)
```bash
export XAI_API_KEY=...
cargo run --example xai
```

### Ollama
Ensure Ollama is running locally (default: http://localhost:11434).
```bash
cargo run --example ollama
```
