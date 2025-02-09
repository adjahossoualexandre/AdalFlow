from typing import (
    Dict,
    Optional,
    Any,
    TypeVar,
    Union,
)
import logging
from adalflow.core.types import ModelType, GeneratorOutput


from adalflow.core.types import EmbedderOutput, CompletionUsage, Embedding
from adalflow.core import ModelClient

from mistralai import Mistral
from mistralai.models import ChatCompletionResponse, CompletionEvent, EmbeddingResponse
from mistralai.utils.eventstreaming import EventStream, EventStreamAsync
import os


log = logging.getLogger(__name__)

T = TypeVar("T")


def parse_stream_response(completion: CompletionEvent) -> str:
    r"""Parse the response of the stream API."""
    return completion.data.choices[0].delta.content


def handle_streaming_response(event_stream: list[CompletionEvent]) -> str:
    r"""Handle the streaming response."""
    full_raw_response = ""
    for completion in event_stream:
        log.debug(f"Raw chunk completion: {completion}")
        raw_response = parse_stream_response(completion)
        full_raw_response += raw_response
    return full_raw_response


def get_event_stream_usage(completion: list[CompletionEvent]) -> dict[str, int]:
    usage_dict = dict(completion_tokens=0, prompt_tokens=0, total_tokens=0)
    usage = completion[-1].data.usage
    usage_dict["completion_tokens"] = usage.completion_tokens
    usage_dict["prompt_tokens"] = usage.prompt_tokens
    usage_dict["total_tokens"] = usage.total_tokens
    return usage_dict


class MistralClient(ModelClient):

    def __init__(
        self, model_name: Optional[str] = None, api_key: Optional[str] = None
    ) -> None:
        super().__init__()
        self._model_name = model_name
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        self.sync_client = self.init_sync_client()
        self.async_client = None

    def init_sync_client(self):
        if not self.api_key:
            raise ValueError("Environment variable MISTRAL_API_KEY must be set")
        return Mistral(api_key=self.api_key)

    def init_async_client(self):
        pass

    def convert_inputs_to_api_kwargs(
        self,
        input: Any,  # for retriever, it is a single query,
        model_kwargs: dict = {},
        model_type: ModelType = ModelType.UNDEFINED,
    ) -> dict:
        api_kwargs = model_kwargs.copy()
        if model_type == ModelType.EMBEDDER:
            if isinstance(input, str):
                input = [input]
            api_kwargs["inputs"] = input
        if model_type == ModelType.LLM:
            assert "model" in api_kwargs, "argument 'model' must be specified"
            api_kwargs["messages"] = [
                {"role": "user", "content": input},
            ]
        return api_kwargs

    def parse_embedding_response(self, response: EmbeddingResponse) -> EmbedderOutput:
        try:
            embeddings = Embedding(embedding=response.data[0].embedding, index=0)
            return EmbedderOutput(data=[embeddings])
        except Exception as e:
            log.error(f"Error parsing the embedding response: {e}")
            return EmbedderOutput(data=[], error=str(e), raw_response=response)

    def parse_chat_completion(
        self, completion: Union[ChatCompletionResponse, EventStream]
    ) -> GeneratorOutput:
        log.debug(f"completion: {completion}")
        if completion is not None:
            try:
                # Handle streamng
                if isinstance(completion, EventStream):  # streaming
                    # convert comlpetion to list to make it reusable. EventStream is a GeneratorType
                    completion: list[CompletionEvent] = list(completion)
                    raw_response = handle_streaming_response(completion)
                else:
                    raw_response = completion.choices[0].message.content
            except Exception as e:
                log.error(f"Error parsing the completion: {e}")
                return GeneratorOutput(
                    data=None, error=str(e), raw_response=str(completion)
                )
            usage: CompletionUsage = self.track_completion_usage(completion)
            return GeneratorOutput(
                data=None, error=None, raw_response=raw_response, usage=usage
            )

    def track_completion_usage(
        self,
        completion: Union[ChatCompletionResponse, list[CompletionEvent]],
    ) -> CompletionUsage:
        if isinstance(completion, ChatCompletionResponse):
            usage: CompletionUsage = CompletionUsage(
                completion_tokens=completion.usage.completion_tokens,
                prompt_tokens=completion.usage.prompt_tokens,
                total_tokens=completion.usage.total_tokens,
            )
            return usage
        else:
            event_stream_usage: dict = get_event_stream_usage(completion)
            usage: CompletionUsage = CompletionUsage(**event_stream_usage)
            return usage

    def call(
        self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED
    ) -> Optional[
        Union[EventStream[CompletionEvent], ChatCompletionResponse, EmbeddingResponse]
    ]:
        if model_type == ModelType.EMBEDDER:
            return self.sync_client.embeddings.create(**api_kwargs)
        if model_type == ModelType.LLM:
            # "stream" as an api_kwargs for consistency with other model clients
            if "stream" in api_kwargs:
                log.debug("streaming call")
                return self.sync_client.chat.stream(**api_kwargs)
            else:
                return self.sync_client.chat.complete(**api_kwargs)

    async def acall(
        self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED
    ) -> Optional[
        Union[
            EventStreamAsync[CompletionEvent], ChatCompletionResponse, EmbeddingResponse
        ]
    ]:
        if "model" not in api_kwargs:
            raise ValueError("model must be specified")
        if model_type == ModelType.EMBEDDER:
            # return await self.sync_client.embeddings.create(**api_kwargs)
            return await self.sync_client.embeddings.create(**api_kwargs)
        if model_type == ModelType.LLM:
            # "stream" as an api_kwargs for consistency with other model clients
            if api_kwargs["stream"]:
                log.debug("streaming call")
                rslt = await self.sync_client.chat.stream_async(**api_kwargs)
                return rslt
            else:
                return await self.sync_client.chat.complete_async(**api_kwargs)

    @classmethod
    def from_dict(cls: type[T], data: Dict[str, Any]) -> T:
        obj = super().from_dict(data)
        # recreate the existing clients
        obj.sync_client = obj.init_sync_client()
        obj.async_client = obj.init_async_client()
        return obj

    def to_dict(self) -> Dict[str, Any]:
        r"""Convert the component to a dictionary."""
        # TODO: not exclude but save yes or no for recreating the clients
        exclude = [
            "sync_client",
            "async_client",
        ]  # unserializable object
        output = super().to_dict(exclude=exclude)
        return output
