import unittest
from unittest.mock import patch, Mock

from mistralai.models import (
    ChatCompletionResponse,
    UsageInfo,
)


from adalflow.core.types import ModelType, GeneratorOutput
from adalflow.components.model_client.mistral_client import MistralClient


class TestMistralModelClient(unittest.TestCase):

    def setUp(self):
        self.client = MistralClient(api_key="fake_api_key")
        self.mock_response = {
            "id": "cmpl-3Q8Z5J9Z1Z5z5",
            "created": 1635820005,
            "object": "chat.completion",
            "model": "open-mistral-nemo",
            "choices": [
                {
                    "message": {
                        "content": "Hello, world!",
                        "role": "assistant",
                    },
                    "index": 0,
                    "finish_reason": "stop",
                }
            ],
            "usage": UsageInfo(completion_tokens=10, prompt_tokens=20, total_tokens=30),
        }
        self.mock_response = ChatCompletionResponse(**self.mock_response)
        self.api_kwargs = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "gpt-3.5-turbo",
        }

    @patch(
        "adalflow.components.model_client.mistral_client.MistralClient.init_sync_client"
    )
    @patch("adalflow.components.model_client.mistral_client.Mistral")
    def test_call(self, MockSyncMistral, mock_init_sync_client):
        mock_sync_client = Mock()
        MockSyncMistral.return_value = mock_sync_client
        mock_init_sync_client.return_value = mock_sync_client

        # Mock the client's api: chat.completions.create

        mock_sync_client.chat.complete = Mock(return_value=self.mock_response)

        # Set the sync client
        self.client.sync_client = mock_sync_client

        # Call the call method
        result = self.client.call(api_kwargs=self.api_kwargs, model_type=ModelType.LLM)

        # Assertions
        mock_sync_client.chat.complete.assert_called_once_with(**self.api_kwargs)
        self.assertEqual(result, self.mock_response)

        # test parse_chat_completion
        output = self.client.parse_chat_completion(completion=self.mock_response)
        self.assertTrue(isinstance(output, GeneratorOutput))
        self.assertEqual(output.raw_response, "Hello, world!")
        self.assertEqual(output.usage.completion_tokens, 10)
        self.assertEqual(output.usage.prompt_tokens, 20)
        self.assertEqual(output.usage.total_tokens, 30)


if __name__ == "__main__":
    unittest.main()

# if __name__ == "__main__":
#    from adalflow.core import Generator
#    from adalflow.core import Embedder
#    from adalflow.utils import setup_env
#    from adalflow.core.string_parser import JsonParser
#
#    setup_env()
#
#    # test generation
#    print("-" * 24)
#    print("Test Generation pipeline")
#    print("-" * 24)
#    rag_template = r"""<START_OF_SYSTEM_MESSAGE>
# You are a helpful assistant.
#
# Your task is to answer the query that may or may not come with context information.
# When context is provided, you should stick to the context and less on your prior knowledge to answer the query.
# <END_OF_SYSTEM_MESSAGE>
# <START_OF_USER_MESSAGE>
#    <START_OF_QUERY>
#    {{input_str}}
#    <END_OF_QUERY>
#    {% if context_str %}
#    <START_OF_CONTEXT>
#    {{context_str}}
#    <END_OF_CONTEXT>
#    {% endif %}
# <END_OF_USER_MESSAGE>
# """
#
#    template = """{{input_str}}"""
#
#    model_kwargs = {
#        "model": "open-mistral-nemo",
#        "temperature": 1,
#        "stream": True,
#    }
#    prompt_kwargs = {
#        "input_str": "What is the capital of France?",
#    }
#
#    mistral_client = MistralClient()
#    generator = Generator(
#        model_client=mistral_client,
#        model_kwargs=model_kwargs,
#        prompt_kwargs=prompt_kwargs,
#        template=template,
#    )
#    output = generator(prompt_kwargs=prompt_kwargs)
#    print(output)
#
#    # test embedding
#    print("-" * 24)
#    print("Test Embedding pipeline")
#    print("-" * 24)
#    model_kwargs.pop("stream")
#    model_kwargs.pop("temperature")
#    model_kwargs["model"] = "mistral-embed"
#    embedder = Embedder(
#        model_client=mistral_client,
#        model_kwargs=model_kwargs,
#    )
#    output = embedder(["where is Brian?"])
#    print(output)
