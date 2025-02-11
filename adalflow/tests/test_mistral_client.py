import unittest
from unittest.mock import patch, Mock, AsyncMock

from mistralai.models import (
    ChatCompletionResponse,
    UsageInfo,
)

from adalflow.core.types import ModelType, GeneratorOutput
from adalflow.components.model_client.mistral_client import MistralClient


def getenv_side_effect(key):
    # This dictionary can hold more keys and values as needed
    env_vars = {"MISTRAL_API_KEY": "fake_api_key"}
    return env_vars.get(key, None)  # Returns None if key is not found


class TestMistralModelClient(unittest.IsolatedAsyncioTestCase):

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

    @patch("adalflow.components.model_client.mistral_client.Mistral")
    async def test_acall_llm(self, MockSyncMistral):
        # the mistralai library has no dedicated async client. It uses the same client for sync and async.
        mock_sync_client = Mock()
        MockSyncMistral.chat.return_value = mock_sync_client

        # Mock the response
        mock_sync_client.chat.complete_async = AsyncMock(
            return_value=self.mock_response
        )

        # Set the sync client
        self.client.sync_client = mock_sync_client
        # Mock the client's api: chat.completions.create
        result = await self.client.acall(
            api_kwargs=self.api_kwargs, model_type=ModelType.LLM
        )

        # Assertions
        mock_sync_client.chat.complete_async.assert_called_once_with(**self.api_kwargs)
        self.assertEqual(result, self.mock_response)

    @patch(
        "adalflow.components.model_client.mistral_client.MistralClient.init_sync_client"
    )
    @patch("adalflow.components.model_client.mistral_client.Mistral")
    def test_call_llm(self, MockSyncMistral, mock_init_sync_client):
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
