import unittest
from unittest.mock import MagicMock, patch
from verbalized_sampling.llms.ollama import OllamaLLM
from verbalized_sampling.llms import get_model
from pydantic import BaseModel

class TestOllamaLLM(unittest.TestCase):
    @patch('verbalized_sampling.llms.ollama.OpenAI')
    def test_init(self, mock_openai):
        config = {"temperature": 0.7}
        llm = OllamaLLM("llama3", config)

        # Verify OpenAI client initialization
        mock_openai.assert_called_with(base_url="http://localhost:11434/v1", api_key="ollama")
        self.assertEqual(llm.model_name, "llama3")
        self.assertEqual(llm.config, config)

    @patch('verbalized_sampling.llms.ollama.OpenAI')
    def test_chat(self, mock_openai):
        # Setup mock
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()

        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.return_value = mock_completion
        mock_completion.choices = [mock_choice]
        mock_choice.message = mock_message
        mock_message.content = "Hello, world!"

        config = {"temperature": 0.7}
        llm = OllamaLLM("llama3", config)

        messages = [{"role": "user", "content": "Hi"}]
        response = llm._chat(messages)

        # Verify call
        mock_client.chat.completions.create.assert_called_with(
            model="llama3",
            messages=messages,
            temperature=0.7
        )
        self.assertEqual(response, "Hello, world!")

    @patch('verbalized_sampling.llms.ollama.OpenAI')
    def test_chat_quoted_response(self, mock_openai):
        # Setup mock
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()

        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.return_value = mock_completion
        mock_completion.choices = [mock_choice]
        mock_choice.message = mock_message
        mock_message.content = "\"Hello, world!\""

        llm = OllamaLLM("llama3", {})
        messages = [{"role": "user", "content": "Hi"}]
        response = llm._chat(messages)

        self.assertEqual(response, "Hello, world!")

    @patch('verbalized_sampling.llms.ollama.OpenAI')
    def test_chat_with_format(self, mock_openai):
        # Setup mock
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()

        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.return_value = mock_completion
        mock_completion.choices = [mock_choice]
        mock_choice.message = mock_message
        # Return a valid JSON response string
        mock_message.content = '{"response": "Positive", "probability": 0.9}'

        class ResponseSchema(BaseModel):
            response: str
            probability: float

        llm = OllamaLLM("llama3", {})
        messages = [{"role": "user", "content": "Classify"}]

        result = llm._chat_with_format(messages, ResponseSchema)

        # Check call arguments
        # Since schema can be transformed, we check if response_format was passed
        _, kwargs = mock_client.chat.completions.create.call_args
        self.assertIn('response_format', kwargs)

        # Check result parsing
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['response'], "Positive")
        self.assertEqual(result[0]['probability'], 0.9)

    def test_get_model_ollama(self):
        # Mocking registry to avoid importing other LLMs which might fail if dependencies missing or need keys
        # But we want to test real get_model logic.
        # We can assume OllamaLLM is registered since we updated __init__.py

        config = {}
        llm = get_model("ollama/llama3", method=None, config=config)

        self.assertIsInstance(llm, OllamaLLM)
        self.assertEqual(llm.model_name, "llama3")

if __name__ == '__main__':
    unittest.main()
