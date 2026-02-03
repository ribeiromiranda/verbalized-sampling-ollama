
<div align="center">
<h1>Verbalized Sampling (Ollama Support Fork)</h1>
<h2>How to Mitigate Mode Collapse and Unlock LLM Diversity</h2>
</div>

---

**Verbalized Sampling (VS)** is a simple prompting strategy that improves LLM diversity by 2-3x. It works by asking the model to generate multiple responses with their probabilities, then sampling from this distribution. VS is **training-free** (works with any LLM via prompting), **model-agnostic** (GPT, Claude, Gemini, Llama, etc.), **orthogonal to temperature**, and effective across tasks like **creative writing**, **social simulation**, **synthetic data generation**, and **open-ended QA**.

**This fork of [Verbalized Sampling](https://github.com/CHATS-lab/verbalized-sampling) adds support for local execution using [Ollama](https://ollama.com/).**

## Quickstart

To try Verbalized Sampling, just copy and paste this into any chatbot (ChatGPT, Claude, Gemini, etc.). For best results, we recommend starting with models like GPT-5, Claude 4 Opus, and Gemini 2.5 Pro:

```
<instructions>
Generate 5 responses to the user query, each within a separate <response> tag. Each <response> must include a <text> and a numeric <probability>.
Please sample at random from the tails of the distribution, such that the probability of each response is less than 0.10.
</instructions>

Tell me a short story about a bear.
```

If you want more stories, just respond and ask `Tell me 5 more stories` in the same conversation. For even better results, paste this into a `system prompt` instead:

```
You are a helpful assistant. For each query, please generate a set of five possible responses, each within a separate <response> tag. Each <response> must include a <text> and a numeric <probability>.
Please sample at random from the tails of the distribution, such that the probability of each response is less than 0.10.
```
For practical tips on getting the most out of this technique and general troubleshooting, please refer to this [X/Twitter thread](https://x.com/dch/status/1978471395173740900)!

## Installation and Usage

To use this fork, install the package from source:

```bash
pip install .
```

## Ollama Support

This fork allows you to run Verbalized Sampling locally using Ollama.

1.  **Install Ollama**: Follow instructions at [ollama.com](https://ollama.com).
2.  **Pull a Model**: Run `ollama pull llama3:8b` (or any other model).
3.  **Run with Python**:

```python
from verbalized_sampling import verbalize

# Generate distribution of responses using a local Ollama model
dist = verbalize(
    "Tell me a joke",
    k=5,
    tau=0.10,
    temperature=0.7,
    provider="ollama",          # Specify ollama provider
    model="llama3:8b",          # Specify your local model name
)

# Sample from the distribution
joke = dist.sample(seed=42)
print(joke.text)
```

You can configure the Ollama base URL by setting the `OLLAMA_BASE_URL` environment variable (default: `http://localhost:11434`).
  

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
