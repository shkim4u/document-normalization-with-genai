import logging
import threading
from typing import List, Dict, Any, Union, Optional, Sequence
from uuid import UUID

from langchain.schema import LLMResult
from langchain_community.callbacks.bedrock_anthropic_callback import BedrockAnthropicTokenUsageCallbackHandler, \
    _get_anthropic_claude_token_cost
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TokenUsageCallbackHandler(BedrockAnthropicTokenUsageCallbackHandler):
    """
    Callback Handler that tracks token usage.
    Refer to:
    - https://api.python.langchain.com/en/latest/callbacks/langchain_core.callbacks.base.BaseCallbackHandler.html
    """
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    successful_requests: int = 0
    total_cost: float = 0.0

    def __init__(self, **kwargs):
        super().__init__()
        log_level = kwargs.get("log_level", "info")
        logger.setLevel(log_level.upper())
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return (
            f"Successful Requests: {self.successful_requests}\n"
            f"Tokens Used: {self.total_tokens}\n"
            f"\tPrompt Tokens: {self.prompt_tokens}\n"
            f"\tCompletion Tokens: {self.completion_tokens}\n"
            f"Total Cost (USD): ${self.total_cost}"
        )

    # def on_llm_start(
    #         self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    # ) -> Any:
    def on_llm_start(self, serialized, prompts: List[str], **kwargs):
        """Run when LLM starts running."""
        logger.info(f"LLM Start: {prompts}")

    def on_chat_model_start(
            self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], **kwargs: Any
    ) -> Any:
        """Run when Chat Model starts running."""
        pass

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        """Run on new LLM token. Only available when streaming is enabled."""
        pass

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """
        Run when LLM ends running.
        We collect token usage here.
        """

        if response.generations is None:
            return super().on_llm_end(response, **kwargs)

        for generation in response.generations:
            logger.info(f"LLM End Message: {generation[0].message.content}")
            token_usage = generation[0].message.response_metadata["usage"]
            completion_tokens = token_usage.get("completion_tokens", 0)
            prompt_tokens = token_usage.get("prompt_tokens", 0)
            total_tokens = token_usage.get("total_tokens", 0)
            model_id = response.llm_output.get("model_id", None)
            total_cost = _get_anthropic_claude_token_cost(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                model_id=model_id,
            )

            # update shared state behind lock
            with self._lock:
                self.total_cost += total_cost
                self.total_tokens += total_tokens
                self.prompt_tokens += prompt_tokens
                self.completion_tokens += completion_tokens
                self.successful_requests += 1

    def on_llm_error(
            self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when LLM errors."""
        pass

    def on_chain_start(
            self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> Any:
        """Run when chain starts running."""
        pass

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs):
        # self.report()
        pass

    def on_chain_error(
            self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when chain errors."""
        pass

    def on_retriever_start(
            self,
            serialized: Dict[str, Any],
            query: str,
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            tags: Optional[List[str]] = None,
            metadata: Optional[Dict[str, Any]] = None,
            **kwargs: Any,
    ) -> Any:
        """Run when Retriever starts running."""
        logger.debug(f"Retriever[RUN_ID={run_id}] Start: {query}")

    def on_retriever_end(
            self,
            documents: Sequence[Document],
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            **kwargs: Any,
    ) -> Any:
        """Run when Retriever ends running."""
        logger.debug(f"Retriever[RUN_ID={run_id}] End: {documents}")

    def clear_report(self):
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.successful_requests = 0
        self.total_cost = 0.0

    def report(self):
        # print(
            # f"\nSuccessful Requests: {self.successful_requests}\n"
            # f"Token Counts:\n"
            # f"Total: {self.total_tokens}\n"
            # f"Prompt: {self.prompt_tokens}\n"
            # f"Completion:{self.completion_tokens}\n"
            # f"Total Cost (USD): ${self.total_cost}\n"
        # )
        print(self.__repr__())
