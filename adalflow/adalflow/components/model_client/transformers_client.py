"""Huggingface transformers ModelClient integration."""

from typing import Any, Dict, Union, List, Optional, Sequence
import logging
from functools import lru_cache
import re
import warnings


from adalflow.core.model_client import ModelClient
from adalflow.core.types import GeneratorOutput, ModelType, Embedding, EmbedderOutput
from adalflow.core.functional import get_top_k_indices_scores

# optional import
from adalflow.utils.lazy_import import safe_import, OptionalPackages


transformers = safe_import(
    OptionalPackages.TRANSFORMERS.value[0], OptionalPackages.TRANSFORMERS.value[1]
)
torch = safe_import(OptionalPackages.TORCH.value[0], OptionalPackages.TORCH.value[1])

import torch

import torch.nn.functional as F
from torch import Tensor

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    pipeline
)

from os import getenv as get_env_variable

log = logging.getLogger(__name__)


def average_pool(last_hidden_states: Tensor, attention_mask: list) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

def mean_pooling(model_output: dict, attention_mask) -> Tensor:
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class TransformerEmbeddingModelClient(ModelClient):

    #
    #   Model initialisation
    #
    def __init__(
            self,
            model_name: Optional[str] = None,
            tokenizer_kwargs: Optional[dict] = dict(),
            auto_model: Optional[type] = AutoModel,
            auto_tokenizer: Optional[type] = AutoTokenizer,
            custom_model: Optional[PreTrainedModel] = None,
            custom_tokenizer: Optional[Union[PreTrainedTokenizer, PreTrainedTokenizerFast]] = None
            ):

        super().__init__()
        self.model_name = model_name
        self.tokenizer_kwargs = tokenizer_kwargs
        if "return_tensors" not in self.tokenizer_kwargs:
            self.tokenizer_kwargs["return_tensors"]= "pt"
        self.auto_model=auto_model
        self.auto_tokenizer=auto_tokenizer
        self.custom_model=custom_model
        self.custom_tokenizer=custom_tokenizer

        # Check if there is conflicting arguments
        self.use_auto_model = auto_model is not None
        self.use_auto_tokenizer = auto_tokenizer is not None
        self.use_cusom_model = custom_model is not None
        self.use_cusom_tokenizer = custom_tokenizer is not None
        self.model_name_exit = model_name is not None

        ## arguments related to model
        if self.use_auto_model and self.use_cusom_model:
            raise ValueError("Cannot specify 'auto_model' and 'custom_model'.")
        elif (not self.use_auto_model) and (not self.use_cusom_model):
            raise ValueError("Need to specify either 'auto_model' or 'custom_model'.")
        elif self.use_auto_model and (not self.model_name_exit):
            raise ValueError("When 'auto_model' is specified 'model_name' must be specified too.")
        
        ## arguments related to tokenizer
        if self.use_auto_tokenizer and self.use_cusom_tokenizer:
            raise Exception("Cannot specify 'auto_tokenizer' and 'custom_tokenizer'.")
        elif (not self.use_auto_tokenizer) and (not self.use_cusom_tokenizer):
            raise Exception("Need to specify either'auto_tokenizer' and 'custom_tokenizer'.")
        elif self.use_auto_tokenizer and (not self.model_name_exit):
            raise ValueError("When 'auto_tokenizer' is specified 'model_name' must be specified too.")

        self.init_sync_client()

    def init_sync_client(self):
        self.init_model(
            model_name=self.model_name,
            auto_model=self.auto_model,
            auto_tokenizer=self.auto_tokenizer,
            custom_model=self.custom_model,
            custom_tokenizer=self.custom_tokenizer
            )

    @lru_cache(None)
    def init_model(
        self,
        model_name: Optional[str] = None,
        auto_model: Optional[type] = AutoModel,
        auto_tokenizer: Optional[type] = AutoTokenizer,
        custom_model: Optional[PreTrainedModel] = None,
        custom_tokenizer: Optional[PreTrainedTokenizer | PreTrainedTokenizerFast] = None
        ):

        try:
            if self.use_auto_model:
                self.model = auto_model.from_pretrained(model_name)
            else:
                self.model = custom_model

            if self.use_auto_tokenizer:
                self.tokenizer = auto_tokenizer.from_pretrained(model_name)
            else:
                self.tokenizer = custom_tokenizer

            log.info(f"Done loading model {model_name}")

        except Exception as e:
            log.error(f"Error loading model {model_name}: {e}")
            raise e

    #
    #   Inference code
    #
    def infer_embedding(
        self,
        input=Union[str, List[str], List[List[str]]],
        tolist: bool = True,
    ) -> Union[List, Tensor]:
        model = self.model

        self.handle_input(input)
        batch_dict = self.tokenize_inputs(input, kwargs=self.tokenizer_kwargs)
        outputs = self.compute_model_outputs(batch_dict, model)
        embeddings = self.compute_embeddings(outputs, batch_dict)

        # normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        if tolist:
            embeddings = embeddings.tolist()
        return embeddings

    def handle_input(self, input: Union[str, List[str], List[List[str]]]) -> Union[List[str], List[List[str]]]:
        if isinstance(input, str):
            input = [input]
        return input
     
    def tokenize_inputs(self, input: Union[str, List[str], List[List[str]]], kwargs: Optional[dict] = dict()) -> dict:
        batch_dict = self.tokenizer(input, **kwargs)
        return batch_dict

    def compute_model_outputs(self, batch_dict: dict, model: PreTrainedModel) -> dict:
        with torch.no_grad():
            outputs = model(**batch_dict)
        return outputs

    def compute_embeddings(self, outputs: dict, batch_dict: dict):
        embeddings = mean_pooling(
            outputs, batch_dict["attention_mask"]
        )
        return embeddings

    #
    # Preprocessing, postprocessing and call for inference code
    #
    def call(self, api_kwargs: Dict = {}, model_type: Optional[ModelType]= ModelType.UNDEFINED) -> Union[List, Tensor]:
        
        if "model" not in api_kwargs:
            raise ValueError("model must be specified in api_kwargs")
        # I don't think it is useful anymore
        # if (
        #     model_type == ModelType.EMBEDDER
        #     # and "model" in api_kwargs
        # ):
        if "mock" in api_kwargs and api_kwargs["mock"]:
            import numpy as np

            embeddings = np.array([np.random.rand(768).tolist()])
            return embeddings

        # inference the model
        return self.infer_embedding(api_kwargs["input"])

    def parse_embedding_response(self, response: Union[List, Tensor]) -> EmbedderOutput:
        embeddings: List[Embedding] = []
        for idx, emb in enumerate(response):
            embeddings.append(Embedding(index=idx, embedding=emb))
        response = EmbedderOutput(data=embeddings)
        return response

    def convert_inputs_to_api_kwargs(
        self,
        input: Any,  # for retriever, it is a single query,
        model_kwargs: dict = {},
        model_type: Optional[ModelType]= ModelType.UNDEFINED
    ) -> dict:
        final_model_kwargs = model_kwargs.copy()
        # if model_type == ModelType.EMBEDDER:
        final_model_kwargs["input"] = input
        return final_model_kwargs


class TransformerLLMModelClient(ModelClient):

    #
    #   Model initialisation
    #
    def __init__(
        self,
        model_name: Optional[str] = None,
        tokenizer_kwargs: Optional[dict] = {},
        init_from: Optional[str] = "autoclass",
        apply_chat_template: bool = False,
        chat_template: Optional[str] = None,
        chat_template_kwargs: Optional[dict] = dict(tokenize=False, add_generation_prompt=True),
        use_token: bool = False,
        torch_dtype: Optional[Any] = torch.bfloat16,
        local_files_only: Optional[bool] = False
    ):
        super().__init__()

        self.model_name = model_name  # current model to use
        self.tokenizer_kwargs = tokenizer_kwargs
        if "return_tensors" not in self.tokenizer_kwargs:
            self.tokenizer_kwargs["return_tensors"]= "pt"
        self.use_token = use_token
        self.torch_dtype = torch_dtype
        self.init_from = init_from
        self.apply_chat_template = apply_chat_template
        self.chat_template = chat_template
        self.chat_template_kwargs = chat_template_kwargs
        self.local_files_only = local_files_only
        self.model = None
        if model_name is not None:
            self.init_model(model_name=model_name)

    def _check_token(self, token: str):
        if get_env_variable(token) is None:
            warnings.warn(
                f"{token} is not set. You may not be able to access the model."
            )

    def _get_token_if_relevant(self) -> Union[str, bool]:
        if self.use_token:
            self._check_token("HF_TOKEN")
            token = get_env_variable("HF_TOKEN")
        else:
            token = False      
        return token

    def _init_from_pipeline(self):

        clean_device_cache()
        token = self._get_token_if_relevant() # return a token string or False
        self.model = pipeline(
            "text-generation",
            model=self.model_name,
            torch_dtype=self.torch_dtype,
            device=get_device(),
            token=token
        )

    def _init_from_automodelcasual_lm(self):

        token = self._get_token_if_relevant() # return a token str or False

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=token,
            local_files_only=self.local_files_only,
            **self.tokenizer_kwargs
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            device_map="auto",
            token=token,
            local_files_only=self.local_files_only
        )
        # Set pad token if it's not already set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token  # common fallback
            self.model.config.pad_token_id = (
                self.tokenizer.eos_token_id
            )  # ensure consistency in the model config


    @lru_cache(None)
    def init_model(self, model_name: str):

        log.debug(f"Loading model {model_name}") 
        try:
            if self.init_from == "autoclass":
                self._init_from_automodelcasual_lm()
            elif self.init_from == "pipeline":
                self._init_from_pipeline()
            else:
                raise ValueError("argument 'init_from' must be one of 'autoclass' or 'pipeline'.")
        except Exception as e:
            log.error(f"Error loading model {model_name}: {e}")
            raise e

    #
    #   Inference code
    #
    def _infer_from_pipeline(
        self,
        *,
        model: str,
        messages: Sequence[Dict[str, str]],
        max_tokens: Optional[int] = None,
        apply_chat_template: bool = False,
        chat_template: Optional[str] = None,
        chat_template_kwargs: Optional[dict] = dict(tokenize=False, add_generation_prompt=True),
        **kwargs,
    ):

        if not self.model:
            self.init_model(model_name=model)

        log.info(
            f"Start to infer model {model}, messages: {messages}, kwargs: {kwargs}"
        )
        #  TO DO: add default values in doc
        final_kwargs = {
            "max_new_tokens": max_tokens or 256,
            "do_sample": True,
            "temperature": kwargs.get("temperature", 0.7),
            "top_k": kwargs.get("top_k", 50),
            "top_p": kwargs.get("top_p", 0.95),
        }
        if apply_chat_template:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=self._get_token_if_relevant(),
                local_files_only=self.local_files_only,
                **self.tokenizer_kwargs
            )
            # Set pad token if it's not already set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token  # common fallback
                self.model.config.pad_token_id = (
                    self.tokenizer.eos_token_id
                )  # ensure consistency in the model config

            model_input = self._handle_input(
                messages,
                apply_chat_template=True,
                chat_template=chat_template,
                chat_template_kwargs=chat_template_kwargs,
                )
        else:
            model_input = self._handle_input(messages)

        outputs = self.model(
            model_input,
            **final_kwargs,
        )
        log.info(f"Outputs: {outputs}")
        return outputs

    def _infer_from_automodelcasual_lm(
        self,
        *,
        model: str,
        messages: Sequence[Dict[str, str]],
        max_tokens: Optional[int] = None,
        max_length: Optional[int] = 8192,  # model-agnostic
        apply_chat_template: bool = False,
        chat_template: Optional[str] = None,
        chat_template_kwargs: Optional[dict] = dict(tokenize=False, add_generation_prompt=True),
        **kwargs,
    ):
        if not self.model:
            self.init_model(model_name=model)

        if apply_chat_template:
            model_input = self._handle_input(
                messages,
                apply_chat_template=True,
                chat_template_kwargs=chat_template_kwargs,
                chat_template=chat_template
                )
        else:
           model_input = self._handle_input(messages) 
        input_ids = self.tokenizer(model_input, return_tensors="pt").to(
            get_device()
        )
        outputs_tokens = self.model.generate(**input_ids, max_length=max_length, max_new_tokens=max_tokens, **kwargs)
        outputs = []
        for output in outputs_tokens:
            outputs.append(self.tokenizer.decode(output))
        return outputs

    def _handle_input(
            self,
            messages: Sequence[Dict[str, str]],
            apply_chat_template: bool = False,
            chat_template_kwargs: dict = None,
            chat_template: Optional[str] = None,
            ) -> str:

        if apply_chat_template:
            if chat_template is not None:
                self.tokenizer.chat_template = chat_template
            prompt = self.tokenizer.apply_chat_template(
                messages, **chat_template_kwargs
            )
            if ("tokenize" in chat_template_kwargs) and (chat_template_kwargs["tokenize"] == True):
                prompt = self.tokenizer.decode(prompt)
                return prompt
            else:
                return prompt
        else:
            text = messages[-1]["content"]
            return text

    def infer_llm(
        self,
        *,
        model: str,
        messages: Sequence[Dict[str, str]],
        max_tokens: Optional[int] = None,
        **kwargs,
    ):

        if self.init_from == "pipeline":
            return self._infer_from_pipeline(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                apply_chat_template=self.apply_chat_template,
                chat_template=self.chat_template,
                chat_template_kwargs=self.chat_template_kwargs,
                **kwargs
            )
        else:
            return self._infer_from_automodelcasual_lm(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                apply_chat_template=self.apply_chat_template,
                chat_template=self.chat_template,
                chat_template_kwargs=self.chat_template_kwargs,
                **kwargs
            )

    #
    # Preprocessing, postprocessing and call for inference code
    #
    def call(self, api_kwargs: Dict = {}, model_type: Optional[ModelType]= ModelType.UNDEFINED):

        if "model" not in api_kwargs:
            raise ValueError("model must be specified in api_kwargs")

        model_name = api_kwargs["model"]
        if (model_name != self.model_name) and (self.model_name is not None):
            # need to update the model_name
            log.warning(f"The model passed in 'model_kwargs' is different that the one that has been previously initialised: Updating model from {self.model_name} to {model_name}.")
            self.model_name = model_name
            self.init_model(model_name=model_name)
        elif (model_name != self.model_name) and (self.model_name is None):
            # need to initialize the model for the first time
            self.model_name = model_name
            self.init_model(model_name=model_name)


        output = self.infer_llm(**api_kwargs)
        return output

    def _parse_chat_completion_from_pipeline(self, completion: Any) -> str:

        text = completion[0]["generated_text"]

        pattern = r"(?<=\|assistant\|>).*"

        match = re.search(pattern, text)

        if match:
            text = match.group().strip().lstrip("\\n")
            return text
        else:
            return ""

    def _parse_chat_completion_from_automodelcasual_lm(self, completion: Any) -> GeneratorOutput:
        print(f"completion: {completion}")
        return completion[0]

    def parse_chat_completion(self, completion: Any) -> str:
        try:
            if self.init_from == "pipeline":
                output = self._parse_chat_completion_from_pipeline(completion)
            else:
                output = self._parse_chat_completion_from_automodelcasual_lm(completion)
            return GeneratorOutput(data=output, raw_response=str(completion))
        except Exception as e:
            log.error(f"Error parsing chat completion: {e}")
            return GeneratorOutput(data=None, raw_response=str(completion), error=e)

    def convert_inputs_to_api_kwargs(
        self,
        input: Any,  # for retriever, it is a single query,
        model_kwargs: dict = {},
        model_type: Optional[ModelType]= ModelType.UNDEFINED
    ) -> dict:
        final_model_kwargs = model_kwargs.copy()
        assert "model" in final_model_kwargs, "model must be specified"
        #messages = [{"role": "system", "content": input}]
        messages = [{"role": "user", "content": input}] # Not sure, but it seems to make more sense
        final_model_kwargs["messages"] = messages
        return final_model_kwargs


class TransformerRerankerModelClient(ModelClient):

    #
    #   Model initialisation
    #
    def __init__(
        self,
        model_name: Optional[str] = None,
        tokenizer_kwargs: Optional[dict] = {},
        local_files_only: Optional[bool] = False
    ):
        self.model_name = model_name
        self.tokenizer_kwargs = tokenizer_kwargs
        if "return_tensors" not in self.tokenizer_kwargs:
            self.tokenizer_kwargs["return_tensors"]= "pt"
        self.local_files_only = local_files_only
        if model_name is not None:
            self.init_model(model_name=model_name)

    def init_model(self, model_name: str):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            local_files_only=self.local_files_only,
            **self.tokenizer_kwargs
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            local_files_only=self.local_files_only
            )
            # Check device availability and set the device
            device = get_device()

            # Move model to the selected device
            self.device = device
            self.model.to(device)
            self.model.eval()
            # register the model
            log.info(f"Done loading model {model_name}")

        except Exception as e:
            log.error(f"Error loading model {model_name}: {e}")
            raise e

    #
    #   Inference code
    #

    def infer_reranker(
        self,
        model: str,
        query: str,
        documents: List[str],
    ) -> List[float]:
        if not self.model:
            self.init_model(model_name=model)
        # convert the query and documents to pair input
        input = [(query, doc) for doc in documents]

        with torch.no_grad():

            inputs = self.tokenizer(
                input,
                **self.tokenizer_kwargs
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            scores = (
                self.model(**inputs, return_dict=True)
                .logits.view(
                    -1,
                )
                .float()
            )
            # apply sigmoid to get the scores
            scores = F.sigmoid(scores)

        scores = scores.tolist()
        return scores

    #
    # Preprocessing, postprocessing and call for inference code
    #
    def call(self, api_kwargs: Dict = {}):

        if "model" not in api_kwargs:
            raise ValueError("model must be specified in api_kwargs")

        model_name = api_kwargs["model"]
        if (model_name != self.model_name) and (self.model_name is not None):
            # need to update the model_name
            log.warning(f"The model passed in 'model_kwargs' is different that the one that has been previously initialised: Updating model from {self.model_name} to {model_name}.")
            self.model_name = model_name
            self.init_model(model_name=model_name)
        elif (model_name != self.model_name) and (self.model_name is None):
            # need to initialize the model for the first time
            self.model_name = model_name
            self.init_model(model_name=model_name)

        assert "query" in api_kwargs, "query is required"
        assert "documents" in api_kwargs, "documents is required"
        assert "top_k" in api_kwargs, "top_k is required"

        top_k = api_kwargs.pop("top_k")
        scores = self.infer_reranker(**api_kwargs)
        top_k_indices, top_k_scores = get_top_k_indices_scores(
            scores, top_k
        )
        log.warning(f"output: ({top_k_indices}, {top_k_scores})")
        return top_k_indices, top_k_scores

    def convert_inputs_to_api_kwargs(
        self,
        input: Any,  # for retriever, it is a single query,
        model_kwargs: dict = {},
        model_type: ModelType = ModelType.UNDEFINED,
    ) -> dict:
        final_model_kwargs = model_kwargs.copy()

        assert "model" in final_model_kwargs, "model must be specified"
        assert "documents" in final_model_kwargs, "documents must be specified"
        assert "top_k" in final_model_kwargs, "top_k must be specified"
        final_model_kwargs["query"] = input
        return final_model_kwargs



# # TODO: provide a standard api for embedding and chat models used in local model SDKs
# class TransformerEmbedder:
#     """Local model SDK for transformers.


#     There are two ways to run transformers:
#     (1) model and then run model inference
#     (2) Pipeline and then run pipeline inference

#     This file demonstrates how to
#     (1) create a torch model inference component:  TransformerEmbedder which equalize to OpenAI(), the SyncAPIClient
#     (2) Convert this model inference component to LightRAG API client: TransformersClient

#     The is now just an exmplary component that initialize a certain model from transformers and run inference on it.
#     It is not tested on all transformer models yet. It might be necessary to write one for each model.

#     References:
#     - transformers: https://huggingface.co/docs/transformers/en/index
#     - thenlper/gte-base model:https://huggingface.co/thenlper/gte-base
#     """

#     models: Dict[str, type] = {}

#     def __init__(self, model_name: Optional[str] = "thenlper/gte-base"):
#         super().__init__()

#         if model_name is not None:
#             self.init_model(model_name=model_name)

#     @lru_cache(None)
#     def init_model(self, model_name: str):
#         try:
#             self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#             self.model = AutoModel.from_pretrained(model_name)
#             # register the model
#             self.models[model_name] = self.model
#             log.info(f"Done loading model {model_name}")

#         except Exception as e:
#             log.error(f"Error loading model {model_name}: {e}")
#             raise e

#     def infer_gte_base_embedding(
#         self,
#         input=Union[str, List[str]],
#         tolist: bool = True,
#     ):
#         model = self.models.get("thenlper/gte-base", None)
#         if model is None:
#             # initialize the model
#             self.init_model("thenlper/gte-base")

#         if isinstance(input, str):
#             input = [input]
#         # Tokenize the input texts
#         batch_dict = self.tokenizer(
#             input, max_length=512, padding=True, truncation=True, return_tensors="pt"
#         )
#         outputs = model(**batch_dict)
#         embeddings = average_pool(
#             outputs.last_hidden_state, batch_dict["attention_mask"]
#         )
#         # (Optionally) normalize embeddings
#         embeddings = F.normalize(embeddings, p=2, dim=1)
#         if tolist:
#             embeddings = embeddings.tolist()
#         return embeddings

#     def __call__(self, **kwargs):
#         if "model" not in kwargs:
#             raise ValueError("model is required")

#         if "mock" in kwargs and kwargs["mock"]:
#             import numpy as np

#             embeddings = np.array([np.random.rand(768).tolist()])
#             return embeddings
#         # load files and models, cache it for the next inference
#         model_name = kwargs["model"]
#         # inference the model
#         if model_name == "thenlper/gte-base":
#             return self.infer_gte_base_embedding(kwargs["input"])
#         else:
#             raise ValueError(f"model {model_name} is not supported")


# def get_device():
#     # Check device availability and set the device
#     if torch.cuda.is_available():
#         device = torch.device("cuda")
#         log.info("Using CUDA (GPU) for inference.")
#     elif torch.backends.mps.is_available():
#         device = torch.device("mps")
#         log.info("Using MPS (Apple Silicon) for inference.")
#     else:
#         device = torch.device("cpu")
#         log.info("Using CPU for inference.")

#     return device


# def clean_device_cache():
#     import torch

#     if torch.has_mps:
#         torch.mps.empty_cache()

#         torch.mps.set_per_process_memory_fraction(1.0)


# class TransformerReranker:
#     __doc__ = r"""Local model SDK for a reranker model using transformers.

#     References:
#     - model: https://huggingface.co/BAAI/bge-reranker-base
#     - paper: https://arxiv.org/abs/2309.07597

#     note:
#     If you are using Macbook M1 series chips, you need to ensure ``torch.device("mps")`` is set.
#     """
#     models: Dict[str, type] = {}

#     def __init__(self, model_name: Optional[str] = "BAAI/bge-reranker-base"):
#         self.model_name = model_name or "BAAI/bge-reranker-base"
#         if model_name is not None:
#             self.init_model(model_name=model_name)

#     def init_model(self, model_name: str):
#         try:
#             self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#             self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
#             # Check device availability and set the device
#             device = get_device()

#             # Move model to the selected device
#             self.device = device
#             self.model.to(device)
#             self.model.eval()
#             # register the model
#             self.models[model_name] = self.model  # TODO: better model registration
#             log.info(f"Done loading model {model_name}")

#         except Exception as e:
#             log.error(f"Error loading model {model_name}: {e}")
#             raise e

#     def infer_bge_reranker_base(
#         self,
#         # input=List[Tuple[str, str]],  # list of pairs of the query and the candidate
#         query: str,
#         documents: List[str],
#     ) -> List[float]:
#         model = self.models.get(self.model_name, None)
#         if model is None:
#             # initialize the model
#             self.init_model(self.model_name)

#         # convert the query and documents to pair input
#         input = [(query, doc) for doc in documents]

#         with torch.no_grad():

#             inputs = self.tokenizer(
#                 input,
#                 padding=True,
#                 truncation=True,
#                 return_tensors="pt",
#                 max_length=512,
#             )
#             inputs = {k: v.to(self.device) for k, v in inputs.items()}
#             scores = (
#                 model(**inputs, return_dict=True)
#                 .logits.view(
#                     -1,
#                 )
#                 .float()
#             )
#             # apply sigmoid to get the scores
#             scores = F.sigmoid(scores)

#         scores = scores.tolist()
#         return scores

#     def __call__(self, **kwargs):
#         r"""Ensure "model" and "input" are in the kwargs."""
#         if "model" not in kwargs:
#             raise ValueError("model is required")

#         # if "mock" in kwargs and kwargs["mock"]:
#         #     import numpy as np

#         #     scores = np.array([np.random.rand(1).tolist()])
#         #     return scores
#         # load files and models, cache it for the next inference
#         model_name = kwargs["model"]
#         # inference the model
#         if model_name == self.model_name:
#             assert "query" in kwargs, "query is required"
#             assert "documents" in kwargs, "documents is required"
#             scores = self.infer_bge_reranker_base(kwargs["query"], kwargs["documents"])
#             return scores
#         else:
#             raise ValueError(f"model {model_name} is not supported")


# class TransformerLLM:
#     __doc__ = r"""Local model SDK for transformers LLM.

#     NOTE:
#         This inference component is only specific to the HuggingFaceH4/zephyr-7b-beta model.

#     The example raw output:
#     # <|system|>
#     # You are a friendly chatbot who always responds in the style of a pirate.</s>
#     # <|user|>
#     # How many helicopters can a human eat in one sitting?</s>
#     # <|assistant|>
#     # Ah, me hearty matey! But yer question be a puzzler! A human cannot eat a helicopter in one sitting, as helicopters are not edible. They be made of metal, plastic, and other materials, not food!


#     References:
#     - model: https://huggingface.co/HuggingFaceH4/zephyr-7b-beta
#     - https://huggingface.co/google/gemma-2b
#     - https://huggingface.co/google/gemma-2-2b

#     """
#     models: Dict[str, type] = {}  # to register the model
#     tokenizer: Dict[str, type] = {}

#     model_to_init_func = {
#         "HuggingFaceH4/zephyr-7b-beta": "use_pipeline",
#         "google/gemma-2-2b": "use_pipeline",
#     }

#     def __init__(
#         self,
#         model_name: Optional[str] = None,
#     ):
#         super().__init__()

#         self.model_name = model_name  # current model to use

#         if model_name is not None and model_name not in self.models:
#             self.init_model(model_name=model_name)

#     def _check_token(self, token: str):
#         import os

#         if os.getenv(token) is None:
#             warnings.warn(
#                 f"{token} is not set. You may not be able to access the model."
#             )

#     def _init_from_pipeline(self, model_name: str):
#         from transformers import pipeline

#         clean_device_cache()
#         self._check_token("HF_TOKEN")
#         try:
#             import os

#             pipe = pipeline(
#                 "text-generation",
#                 model=model_name,
#                 torch_dtype=torch.bfloat16,
#                 device=get_device(),
#                 token=os.getenv("HF_TOKEN"),
#             )
#             self.models[model_name] = pipe
#         except Exception as e:
#             log.error(f"Error loading model {model_name}: {e}")
#             raise e

#     def _init_from_automodelcasual_lm(self, model_name: str):
#         try:
#             from transformers import AutoTokenizer, AutoModelForCausalLM
#         except ImportError:
#             raise ImportError(
#                 "transformers is not installed. Please install it with `pip install transformers`"
#             )

#         try:
#             import os

#             if os.getenv("HF_TOKEN") is None:
#                 warnings.warn(
#                     "HF_TOKEN is not set. You may not be able to access the model."
#                 )

#             tokenizer = AutoTokenizer.from_pretrained(
#                 model_name, token=os.getenv("HF_TOKEN")
#             )
#             model = AutoModelForCausalLM.from_pretrained(
#                 model_name,
#                 torch_dtype=torch.bfloat16,
#                 device_map="auto",
#                 token=os.getenv("HF_TOKEN"),
#             )
#             self.models[model_name] = model
#             self.tokenizer[model_name] = tokenizer
#         except Exception as e:
#             log.error(f"Error loading model {model_name}: {e}")
#             raise e

#     @lru_cache(None)
#     def init_model(self, model_name: str):
#         log.debug(f"Loading model {model_name}")

#         model_setup = self.model_to_init_func.get(model_name, None)
#         if model_setup:
#             if model_setup == "use_pipeline":
#                 self._init_from_pipeline(model_name)
#             else:
#                 self._init_from_automodelcasual_lm(model_name)
#         else:
#             raise ValueError(f"Model {model_name} is not supported")

#     def _parse_chat_completion_from_pipeline(self, completion: Any) -> str:

#         text = completion[0]["generated_text"]

#         pattern = r"(?<=\|assistant\|>).*"

#         match = re.search(pattern, text)

#         if match:
#             text = match.group().strip().lstrip("\\n")
#             return text
#         else:
#             return ""

#     def _parse_chat_completion_from_automodelcasual_lm(self, completion: Any) -> str:
#         print(f"completion: {completion}")
#         return completion[0]

#     def parse_chat_completion(self, completion: Any) -> str:
#         model_name = self.model_name
#         model_setup = self.model_to_init_func.get(model_name, None)
#         if model_setup:
#             if model_setup == "use_pipeline":
#                 return self._parse_chat_completion_from_pipeline(completion)
#             else:
#                 return self._parse_chat_completion_from_automodelcasual_lm(completion)
#         else:
#             raise ValueError(f"Model {model_name} is not supported")

#     def _infer_from_pipeline(
#         self,
#         *,
#         model: str,
#         messages: Sequence[Dict[str, str]],
#         max_tokens: Optional[int] = None,
#         **kwargs,
#     ):
#         if not model:
#             raise ValueError("Model is not provided.")

#         if model not in self.models:
#             self.init_model(model_name=model)

#         model_to_use = self.models[model]

#         log.info(
#             f"Start to infer model {model}, messages: {messages}, kwargs: {kwargs}"
#         )

#         if model == "HuggingFaceH4/zephyr-7b-beta":

#             prompt = model_to_use.tokenizer.apply_chat_template(
#                 messages, tokenize=False, add_generation_prompt=True
#             )

#             final_kwargs = {
#                 "max_new_tokens": max_tokens or 256,
#                 "do_sample": True,
#                 "temperature": kwargs.get("temperature", 0.7),
#                 "top_k": kwargs.get("top_k", 50),
#                 "top_p": kwargs.get("top_p", 0.95),
#             }
#             outputs = model_to_use(prompt, **final_kwargs)
#         elif model == "google/gemma-2-2b":
#             final_kwargs = {
#                 "max_new_tokens": max_tokens or 256,
#                 "do_sample": True,
#                 "temperature": kwargs.get("temperature", 0.7),
#                 "top_k": kwargs.get("top_k", 50),
#                 "top_p": kwargs.get("top_p", 0.95),
#             }
#             text = messages[0]["content"]
#             outputs = model_to_use(
#                 text,
#                 **final_kwargs,
#             )

#         log.info(f"Outputs: {outputs}")
#         return outputs

#     def _infer_from_automodelcasual_lm(
#         self,
#         *,
#         model: str,
#         messages: Sequence[Dict[str, str]],
#         max_length: Optional[int] = 8192,  # model-agnostic
#         **kwargs,
#     ):
#         if not model:
#             raise ValueError("Model is not provided.")
#         if model not in self.models:
#             self.init_model(model_name=model)
#         model_to_use = self.models[model]
#         tokenizer_to_use = self.tokenizer[model]

#         input_ids = tokenizer_to_use(messages[0]["content"], return_tensors="pt").to(
#             get_device()
#         )
#         print(input_ids)
#         outputs_tokens = model_to_use.generate(**input_ids, max_length=max_length)
#         outputs = []
#         for i, output in enumerate(outputs_tokens):
#             outputs.append(tokenizer_to_use.decode(output))
#         return outputs

#     def infer_llm(
#         self,
#         *,
#         model: str,
#         messages: Sequence[Dict[str, str]],
#         max_tokens: Optional[int] = None,
#         **kwargs,
#     ):
#         # TODO: generalize the code for more models
#         model_setup = self.model_to_init_func.get(model, None)
#         if model_setup:
#             if model_setup == "use_pipeline":
#                 return self._infer_from_pipeline(
#                     model=model, messages=messages, max_tokens=max_tokens, **kwargs
#                 )
#             else:
#                 return self._infer_from_automodelcasual_lm(
#                     model=model, messages=messages, max_tokens=max_tokens, **kwargs
#                 )
#         else:
#             raise ValueError(f"Model {model} is not supported")

#     def __call__(self, **kwargs):
#         r"""Ensure "model" and "input" are in the kwargs."""
#         log.debug(f"kwargs: {kwargs}")
#         if "model" not in kwargs:
#             raise ValueError("model is required")

#         if "messages" not in kwargs:
#             raise ValueError("messages is required")

#         model_name = kwargs["model"]
#         if model_name != self.model_name:
#             # need to initialize the model and update the model_name
#             self.model_name = model_name
#             self.init_model(model_name=model_name)

#         output = self.infer_llm(**kwargs)
#         return output


# class TransformersClient(ModelClient):
#     __doc__ = r"""LightRAG API client for transformers.

#     Use: ``ls ~/.cache/huggingface/hub `` to see the cached models.

#     Some modeles are gated, you will need to their page to get the access token.
#     Find how to apply tokens here: https://huggingface.co/docs/hub/security-tokens
#     Once you have a token and have access, put the token in the environment variable HF_TOKEN.
#     """

#     support_models = {
#         "thenlper/gte-base": {
#             "type": ModelType.EMBEDDER,
#         },
#         "BAAI/bge-reranker-base": {
#             "type": ModelType.RERANKER,
#         },
#         "HuggingFaceH4/zephyr-7b-beta": {"type": ModelType.LLM},
#         "google/gemma-2-2b": {"type": ModelType.LLM},
#     }

#     def __init__(self, model_name: Optional[str] = None) -> None:
#         super().__init__()
#         self._model_name = model_name
#         if self._model_name:
#             assert (
#                 self._model_name in self.support_models
#             ), f"model {self._model_name} is not supported"
#         if self._model_name == "thenlper/gte-base":
#             self.sync_client = self.init_sync_client()
#         elif self._model_name == "BAAI/bge-reranker-base":
#             self.reranker_client = self.init_reranker_client()
#         elif self._model_name == "HuggingFaceH4/zephyr-7b-beta":
#             self.llm_client = self.init_llm_client()
#         self.async_client = None

#     def init_sync_client(self):
#         return TransformerEmbedder()

#     def init_reranker_client(self):
#         return TransformerReranker()

#     def init_llm_client(self):
#         return TransformerLLM()

#     def set_llm_client(self, llm_client: object):
#         r"""Allow user to pass a custom llm client. Here is an example of a custom llm client:

#         Ensure you have parse_chat_completion and __call__ methods which will be applied to api_kwargs specified in transform_client.call().

#         .. code-block:: python

#             class CustomizeLLM:

#                 def __init__(self) -> None:
#                     pass

#                 def parse_chat_completion(self, completion: Any) -> str:
#                     return completion

#                 def __call__(self, messages: Sequence[Dict[str, str]], model: str, **kwargs):
#                     from transformers import AutoTokenizer, AutoModelForCausalLM

#                     tokenizer = AutoTokenizer.from_pretrained(
#                         "deepseek-ai/deepseek-coder-1.3b-instruct", trust_remote_code=True
#                     )
#                     model = AutoModelForCausalLM.from_pretrained(
#                         "deepseek-ai/deepseek-coder-1.3b-instruct",
#                         trust_remote_code=True,
#                         torch_dtype=torch.bfloat16,
#                     ).to(get_device())
#                     messages = [
#                         {"role": "user", "content": "write a quick sort algorithm in python."}
#                     ]
#                     inputs = tokenizer.apply_chat_template(
#                         messages, add_generation_prompt=True, return_tensors="pt"
#                     ).to(model.device)
#                     # tokenizer.eos_token_id is the id of <|EOT|> token
#                     outputs = model.generate(
#                         inputs,
#                         max_new_tokens=512,
#                         do_sample=False,
#                         top_k=50,
#                         top_p=0.95,
#                         num_return_sequences=1,
#                         eos_token_id=tokenizer.eos_token_id,
#                     )
#                     print(
#                         tokenizer.decode(outputs[0][len(inputs[0]) :], skip_special_tokens=True)
#                     )
#                     decoded_outputs = []
#                     for output in outputs:
#                         decoded_outputs.append(
#                             tokenizer.decode(output[len(inputs[0]) :], skip_special_tokens=True)
#                         )
#                     return decoded_outputs

#             llm_client = CustomizeLLM()
#             transformer_client.set_llm_client(llm_client)
#             # use in the generator
#             generator = Generator(
#                 model_client=transformer_client,
#                 model_kwargs=model_kwargs,
#                 prompt_kwargs=prompt_kwargs,
#                 ...)

#         """
#         self.llm_client = llm_client

#     def parse_embedding_response(self, response: Any) -> EmbedderOutput:
#         embeddings: List[Embedding] = []
#         for idx, emb in enumerate(response):
#             embeddings.append(Embedding(index=idx, embedding=emb))
#         response = EmbedderOutput(data=embeddings)
#         return response

#     def parse_chat_completion(self, completion: Any) -> GeneratorOutput:
#         try:
#             output = self.llm_client.parse_chat_completion(completion)

#             return GeneratorOutput(data=output, raw_response=str(completion))
#         except Exception as e:
#             log.error(f"Error parsing chat completion: {e}")
#             return GeneratorOutput(data=None, raw_response=str(completion), error=e)

#     def call(self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED):
#         if "model" not in api_kwargs:
#             raise ValueError("model must be specified in api_kwargs")
#         if api_kwargs["model"] not in self.support_models:
#             raise ValueError(f"model {api_kwargs['model']} is not supported")

#         if (
#             model_type == ModelType.EMBEDDER
#             and "model" in api_kwargs
#             and api_kwargs["model"] == "thenlper/gte-base"
#         ):
#             if self.sync_client is None:
#                 self.sync_client = self.init_sync_client()
#             return self.sync_client(**api_kwargs)
#         elif (  # reranker
#             model_type == ModelType.RERANKER
#             and "model" in api_kwargs
#             and api_kwargs["model"] == "BAAI/bge-reranker-base"
#         ):
#             if not hasattr(self, "reranker_client") or self.reranker_client is None:
#                 self.reranker_client = self.init_reranker_client()
#             scores = self.reranker_client(**api_kwargs)
#             top_k_indices, top_k_scores = get_top_k_indices_scores(
#                 scores, api_kwargs["top_k"]
#             )
#             return top_k_indices, top_k_scores
#         elif model_type == ModelType.LLM and "model" in api_kwargs:  # LLM
#             if not hasattr(self, "llm_client") or self.llm_client is None:
#                 self.llm_client = self.init_llm_client()
#             response = self.llm_client(**api_kwargs)
#             return response
#         else:
#             raise ValueError(f"model_type {model_type} is not supported")

#     def convert_inputs_to_api_kwargs(
#         self,
#         input: Any,  # for retriever, it is a single query,
#         model_kwargs: dict = {},
#         model_type: ModelType = ModelType.UNDEFINED,
#     ) -> dict:
#         final_model_kwargs = model_kwargs.copy()
#         if model_type == ModelType.EMBEDDER:
#             final_model_kwargs["input"] = input
#             return final_model_kwargs
#         elif model_type == ModelType.RERANKER:
#             assert "model" in final_model_kwargs, "model must be specified"
#             assert "documents" in final_model_kwargs, "documents must be specified"
#             assert "top_k" in final_model_kwargs, "top_k must be specified"
#             final_model_kwargs["query"] = input
#             return final_model_kwargs
#         elif model_type == ModelType.LLM:
#             assert "model" in final_model_kwargs, "model must be specified"
#             messages = [{"role": "system", "content": input}]
#             final_model_kwargs["messages"] = messages
#             return final_model_kwargs
#         else:
#             raise ValueError(f"model_type {model_type} is not supported")


# if __name__ == "__main__":
#     from adalflow.core import Generator

#     import adalflow as adal

#     adal.setup_env()

#     rag_template = r"""<START_OF_SYSTEM_MESSAGE>
# You are a helpful assistant.

# Your task is to answer the query that may or may not come with context information.
# When context is provided, you should stick to the context and less on your prior knowledge to answer the query.
# <END_OF_SYSTEM_MESSAGE>
# <START_OF_USER_MESSAGE>
#     <START_OF_QUERY>
#     {{input_str}}
#     <END_OF_QUERY>
#     {% if context_str %}
#     <START_OF_CONTEXT>
#     {{context_str}}
#     <END_OF_CONTEXT>
#     {% endif %}
# <END_OF_USER_MESSAGE>
# """

#     template = """{{input_str}}"""

#     model_kwargs = {
#         "model": "google/gemma-2-2b",
#         "temperature": 1,
#         "stream": False,
#     }
#     prompt_kwargs = {
#         "input_str": "Where is Brian?",
#         # "context_str": "Brian is in the kitchen.",
#     }
#     prompt_kwargs = {
#         "input_str": "What is the capital of France?",
#     }

#     class CustomizeLLM:

#         def __init__(self) -> None:
#             pass

#         def parse_chat_completion(self, completion: Any) -> str:
#             return completion[0]

#         def __call__(self, messages: Sequence[Dict[str, str]], model: str, **kwargs):
#             r"""take api key"""
#             from transformers import AutoTokenizer, AutoModelForCausalLM

#             tokenizer = AutoTokenizer.from_pretrained(
#                 "deepseek-ai/deepseek-coder-1.3b-instruct", trust_remote_code=True
#             )
#             model = AutoModelForCausalLM.from_pretrained(
#                 "deepseek-ai/deepseek-coder-1.3b-instruct",
#                 trust_remote_code=True,
#                 torch_dtype=torch.bfloat16,
#             ).to(get_device())
#             messages = [
#                 {"role": "user", "content": "write a quick sort algorithm in python."}
#             ]
#             inputs = tokenizer.apply_chat_template(
#                 messages, add_generation_prompt=True, return_tensors="pt"
#             ).to(model.device)
#             # tokenizer.eos_token_id is the id of <|EOT|> token
#             outputs = model.generate(
#                 inputs,
#                 max_new_tokens=512,
#                 do_sample=False,
#                 top_k=50,
#                 top_p=0.95,
#                 num_return_sequences=1,
#                 eos_token_id=tokenizer.eos_token_id,
#             )

#             decoded_outputs = []
#             for output in outputs:
#                 decoded_outputs.append(
#                     tokenizer.decode(output[len(inputs[0]) :], skip_special_tokens=True)
#                 )
#             return decoded_outputs

#     transformer_client = TransformersClient()
#     transformer_client.set_llm_client(CustomizeLLM())
#     generator = Generator(
#         model_client=transformer_client,
#         model_kwargs=model_kwargs,
#         # prompt_kwargs=prompt_kwargs,
#         template=template,
#         # output_processors=JsonParser(),
#     )

#     output = generator(prompt_kwargs=prompt_kwargs)
#     print(output)
