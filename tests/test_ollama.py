import unittest
from unittest.mock import MagicMock, patch
import os
import json
from verbalized_sampling.llms.ollama import OllamaLLM
from verbalized_sampling.llms import get_model
from pydantic import BaseModel


class TestOllamaLLM(unittest.TestCase):
    @patch("verbalized_sampling.llms.ollama.Client")
    def test_init(self, mock_client_cls):
        # Ensure environment doesn't interfere
        with patch.dict(os.environ, {}, clear=True):
            config = {"temperature": 0.7}
            llm = OllamaLLM("llama3", config)

            # Verify Ollama Client initialization with default URL
            mock_client_cls.assert_called_with(host="http://localhost:11434")
            self.assertEqual(llm.model_name, "llama3")
            self.assertEqual(llm.config, config)

    @patch("verbalized_sampling.llms.ollama.Client")
    @patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://custom-url:1234"})
    def test_init_custom_url(self, mock_client_cls):
        config = {"temperature": 0.7}
        llm = OllamaLLM("llama3", config)

        # Verify Ollama Client initialization with custom URL
        mock_client_cls.assert_called_with(host="http://custom-url:1234")

    @patch("verbalized_sampling.llms.ollama.Client")
    def test_chat(self, mock_client_cls):
        # Setup mock
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        # Mock response structure from native client
        mock_client.chat.return_value = {
            "model": "llama3",
            "created_at": "...time...",
            "message": {"role": "assistant", "content": "Hello, world!"},
            "done": True,
        }

        config = {"temperature": 0.7}
        llm = OllamaLLM("llama3", config)

        messages = [{"role": "user", "content": "Hi"}]
        response = llm._chat(messages)

        # Verify call
        mock_client.chat.assert_called_with(
            model="llama3", messages=messages, options={"temperature": 0.7}
        )
        self.assertEqual(response, "Hello, world!")

    @patch("verbalized_sampling.llms.ollama.Client")
    def test_chat_quoted_response(self, mock_client_cls):
        # Setup mock
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_client.chat.return_value = {
            "message": {"role": "assistant", "content": '"Hello, world!"'},
        }

        llm = OllamaLLM("llama3", {})
        messages = [{"role": "user", "content": "Hi"}]
        response = llm._chat(messages)

        self.assertEqual(response, "Hello, world!")

    @patch("verbalized_sampling.llms.ollama.Client")
    def test_chat_newline_preservation(self, mock_client_cls):
        # Setup mock
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        # Response with newlines
        mock_client.chat.return_value = {
            "message": {"role": "assistant", "content": "Line 1\nLine 2"},
        }

        llm = OllamaLLM("llama3", {})
        messages = [{"role": "user", "content": "Hi"}]
        response = llm._chat(messages)

        # Assert newlines are PRESERVED
        self.assertEqual(response, "Line 1\nLine 2")

    @patch("verbalized_sampling.llms.ollama.Client")
    def test_chat_with_format(self, mock_client_cls):
        # Setup mock
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        # Return a valid JSON response string
        mock_client.chat.return_value = {
            "message": {
                "role": "assistant",
                "content": '{"response": "Positive", "probability": 0.9}',
            },
        }

        class ResponseSchema(BaseModel):
            response: str
            probability: float

        llm = OllamaLLM("llama3", {})
        messages = [{"role": "user", "content": "Classify"}]

        result = llm._chat_with_format(messages, ResponseSchema)

        # Check call arguments
        _, kwargs = mock_client.chat.call_args
        self.assertIn("format", kwargs)
        # We can check specific format if needed, but existence is key

        # Check result parsing
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["response"], "Positive")
        self.assertEqual(result[0]["probability"], 0.9)

    def test_parse_response_markdown(self):
        # We don't need to mock Client for this, just test logic which is static-ish
        # Instantiating OllamaLLM will try to create Client, which might fail if we don't mock or handle it.
        # But we can patch Client in setUp or here just to be safe.
        with patch("verbalized_sampling.llms.ollama.Client"):
            llm = OllamaLLM("llama3", {})

            # Markdown wrapped JSON
            raw_json = json.dumps({"responses": [{"text": "bar", "probability": 0.8}]})
            markdown_content = f"Here is the JSON:\n```json\n{raw_json}\n```"

            parsed = llm._parse_response_with_schema(markdown_content)

            self.assertEqual(len(parsed), 1)
            self.assertEqual(parsed[0]["response"], "bar")

    def test_get_model_ollama(self):
        # Mocking registry to avoid importing other LLMs which might fail if dependencies missing or need keys
        config = {}
        # We need to ensure that get_model returns OllamaLLM and it initializes correctly.
        # But we must mock Client inside OllamaLLM init otherwise it tries to connect/init.
        with patch("verbalized_sampling.llms.ollama.Client"):
            llm = get_model("ollama/llama3", method=None, config=config)
            self.assertIsInstance(llm, OllamaLLM)
            self.assertEqual(llm.model_name, "llama3")


if __name__ == "__main__":
    unittest.main()
