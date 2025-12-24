import asyncio
import json
from pathlib import Path
from typing import Dict, Any, Callable, TypeVar, Optional, Mapping, List
import importlib.resources

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel as LangChainModel
    
from browser_use.llm.base import BaseChatModel
from browser_use.llm.openai.chat import ChatOpenAI
from browser_use.llm.messages import UserMessage

COST_MAP = None

# singleton cost map
# https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json
def load_cost_map() -> Dict:
    global COST_MAP
    if not COST_MAP:
        cost_map_text = importlib.resources.files("bupp").joinpath("model_api_prices.json").read_text(encoding="utf-8")
        COST_MAP = json.loads(cost_map_text)

    return COST_MAP

T = TypeVar("T")

def log_messages(
    chat_logdir: Path | None, 
    messages: list[BaseMessage | dict | Any],
    model_name: str
) -> None:
    """Logs the chat message to a file"""
    if chat_logdir is None:
        return

    chat_logdir.mkdir(parents=True, exist_ok=True)

    model_file = chat_logdir / f"model.txt"
    with open(model_file, "w", encoding="utf-8") as f:
        f.write(model_name)

    # Find the next available log file number
    counter = 1
    while True:
        log_file = chat_logdir / f"{counter}.txt"
        if not log_file.exists():
            break
        counter += 1

    with open(log_file, "w", encoding="utf-8") as f:
        message_parts = []
        for msg in messages:
            if isinstance(msg, BaseMessage):
                message_parts.append(f"[{msg.type}] {msg.content}")
            elif isinstance(msg, dict):
                msg_type = msg.get("type", "unknown")
                msg_content = msg.get("content", str(msg))
                message_parts.append(f"[{msg_type}] {msg_content}")
            else:
                message_parts.append(f"[unknown] {str(msg)}")

        print(f"Writing messages to {log_file}")    
    
        concatenated_messages = "\n\n".join(message_parts)
        f.write(concatenated_messages)

def str_to_messages(s: str) -> list[BaseMessage]:
    return [HumanMessage(content=s)]

class ChatModelWithLogging(ChatOpenAI):
    def __init__(self, chat_logdir: Path, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.chat_logdir = chat_logdir

    async def ainvoke(self, messages: list[BaseMessage], output_format: None = None) -> Any: # type: ignore
        res = await super().ainvoke(messages, output_format)
        log_messages(self.chat_logdir, messages, self.model)
        return res

class ChatModelWithName(BaseChatModel):
    """Wrapper for BaseChatModel that adds a model_name attribute."""

    def __init__(self, model: LangChainModel, model_name: str):
        self._model = model
        # Align with BaseChatModel Protocol: establish a public `model` name
        self.model_name = model_name
        self.log_fn: Optional[Callable[[float, str], None]] = None
        self.function_name: str = ""

        self._cost_map = load_cost_map()
        self._cost = 0.0

        # Per-function chat log directory (set by LLMHub)
        self._chat_logdir: Optional[Path] = None

    # Align with BaseChatModel Protocol
    _verified_api_keys: bool = False

    @property
    def chat_logdir(self) -> Path | None:
        return self._chat_logdir

    @property
    def provider(self) -> str:
        mod = self._model.__class__.__module__
        # Best-effort provider inference from module path
        if "openai" in mod:
            return "openai"
        if "anthropic" in mod:
            return "anthropic"
        if "google" in mod:
            return "google"
        if "together" in mod:
            return "together"
        if "xai" in mod:
            return "xai"
        return self._model.__class__.__name__.lower()

    @property
    def name(self) -> str:
        return self.model

    @property
    def model_name(self) -> str:
        # for legacy support
        return self.model
    
    @model_name.setter
    def model_name(self, value: str) -> None:
        self.model = value

    def get_cost(self) -> float:
        return self._cost

    def set_log_fn(
        self,
        log_fn: Callable[[float, str], None],
        function_name: str,
        chat_logdir: Optional[Path] = None,
    ) -> None:
        self.log_fn = log_fn
        self.function_name = function_name
        self._chat_logdir = chat_logdir

    def log_cost(self, res: Any) -> None:
        metadata = getattr(res, "usage_metadata", None)
        if not metadata:
            return
        invoke_cost = 0.0
        invoke_cost += self._cost_map[self.model_name]["input_cost_per_token"] * metadata.get("input_tokens", 0)
        invoke_cost += self._cost_map[self.model_name]["output_cost_per_token"] * metadata.get("output_tokens", 0)
        self._cost += invoke_cost

        if self.log_fn:
            self.log_fn(invoke_cost, self.function_name)

    def invoke(self, prompt: str) -> Any:
        # model = ChatOpenAI(model=self.model_name)
        # if not isinstance(prompt, str):
        #     raise ValueError(f"Message must be a string, got {type(prompt)}")

        # log_messages(self.chat_logdir, str_to_messages(prompt), self.model)

        # # kind of dumb but the calling code has serial dependencies on sync calls
        # res = asyncio.run(model.ainvoke(prompt))
        # self.log_cost(res)
        # return res
        raise Exception("sync calls no longer supported")

    async def ainvoke(self, messages: Any, output_format: Any | None = None) -> Any: # type: ignore
        model = ChatOpenAI(model=self.model_name) 
        if isinstance(messages, str):
            messages = [UserMessage(content=messages)]
        elif isinstance(messages, list):
            messages = [self._serialize_message(msg) for msg in messages]
        else:
            raise ValueError(f"Invalid messages type: {type(messages)}")

        serialized_messages = messages
        log_messages(self.chat_logdir, serialized_messages, self.model)

        # Support the Protocol's signature while remaining permissive for existing call sites
        # if output_format is not None:
        #     model = self._model.with_structured_output(output_format)
        #     res = await model.ainvoke(serialized_messages)
        # else:
        # print("SERIALIZED MESSAGES: ", serialized_messages)
        res = await model.ainvoke(serialized_messages)

        if hasattr(res, "usage_metadata"):
            self.log_cost(res)
        return res

    def __getattr__(self, name: str) -> Any:
        # Delegate all other attribute access to the wrapped model
        return getattr(self._model, name)

    def _serialize_message(self, msg: Any) -> BaseMessage | Any:
        if isinstance(msg, str):
            return UserMessage(content=msg)
        if not isinstance(msg, BaseMessage) and not isinstance(msg, dict):
            return HumanMessage(content=getattr(msg, "content"))
        return msg

# Lazy-init models
def gemini_25_flash():
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatModelWithName(
        ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            api_key=os.getenv("GEMINI_API_KEY"),
            enable_thinking=True
        ),
        "gemini-2.5-flash"
    )

def gemini_25_pro():
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatModelWithName(
        ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            api_key=os.getenv("GEMINI_API_KEY")
        ),
        "gemini-2.5-pro"
    )

def openai_o3_mini():
    from langchain_openai import ChatOpenAI
    return ChatModelWithName(
        ChatOpenAI(model="o3-mini"),
        "o3-mini"
    )

def openai_o4_mini():
    from langchain_openai import ChatOpenAI
    return ChatModelWithName(
        ChatOpenAI(model="o4-mini"),
        "o4-mini"
    )

def openai_4o():
    from langchain_openai import ChatOpenAI
    return ChatModelWithName(
        ChatOpenAI(model="gpt-4o"),
        "gpt-4o"
    )

def openai_41():
    from langchain_openai import ChatOpenAI
    return ChatModelWithName(
        ChatOpenAI(model="gpt-4.1"),
        "gpt-4.1"
    )

def openai_5():
    from langchain_openai import ChatOpenAI
    return ChatModelWithName(
        ChatOpenAI(model="gpt-5"),
        "gpt-5"
    )

def openai_51():
    from langchain_openai import ChatOpenAI
    return ChatModelWithName(
        ChatOpenAI(model="gpt-5.1"),
        "gpt-5.1"
    )

# def grok4():
#     from langchain_xai import ChatXAI
#     return ChatModelWithName(
#         ChatXAI(
#             model="grok-4", 
#             api_key=os.getenv("XAI_API_KEY")
#         ),
#         "grok-4"
#     )

# def cohere_command_a():
#     from langchain_cohere import ChatCohere
#     return ChatModelWithName(
#         ChatCohere(
#             model="command-a-03-2025", 
#             cohere_api_key=os.getenv("COHERE_API_KEY")
#         ),
#         "command-a-03-2025"
#     )

# def together_deepseek_r1():
#     from langchain_together import ChatTogether
#     return ChatModelWithName(
#         ChatTogether(
#             model="deepseek-ai/DeepSeek-R1-0528-tput",
#             api_key=os.getenv("TOGETHER_API_KEY")
#         ),
#         "deepseek-ai/DeepSeek-R1-0528-tput"
#     )

def openai_o3():
    from langchain_openai import ChatOpenAI
    return ChatModelWithName(
        ChatOpenAI(model="o3"),
        "o3"
    )

# def anthropic_claude_3_5_sonnet():
#     from langchain_anthropic import ChatAnthropic
#     return ChatModelWithName(
#         ChatAnthropic(model="claude-3-5-sonnet-20240620"),
#         "claude-3-5-sonnet-20240620"
#     )

# def claude_4_sonnet():
#     from langchain_anthropic import ChatAnthropic
#     return ChatModelWithName(
#         ChatAnthropic(model="claude-sonnet-4-20250514", max_tokens=15000),
#         "claude-sonnet-4-20250514"
#     )

LLM_MODELS = {
    # "command-a-03-2025": cohere_command_a,
    # "deepseeks_r1": together_deepseek_r1,
    "gpt-4o": openai_4o,
    "gpt-4.1": openai_41,
    "gemini-2.5-flash": gemini_25_flash,
    "gemini-2.5-pro": gemini_25_pro,
    # "default": cohere_command_a,
    "o4-mini": openai_o4_mini,
    "o3": openai_o3,
    "o3-mini": openai_o3_mini,
    # "claude-3-5-sonnet-20240620": anthropic_claude_3_5_sonnet,
    # "claude-sonnet-4-20250514": claude_4_sonnet,
    "gpt-5": openai_5,
    "gpt-5.1": openai_51,
}

# incredibly dumb hack to appease the type checker
class BaseChatWrapper:
    def __init__(
            self, 
            function_name: str, 
            model: LangChainModel, 
            log_fn: Callable[[BaseMessage, str], None]
        ):
        self._function_name = function_name
        self._model = model
        self._log_fn = log_fn

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        res = self._model.invoke(*args, **kwargs)
        self._log_fn(res, self._function_name)
        return res

class LLMHub:
    """
    Thin convenience wrapper around a collection of LangChain chat models.

    Args:
        providers (Dict[str, BaseChatModel]):
            Mapping of model-name → model instance.
        function_map (Dict[str, str]):
            Mapping of function-name -> model-name.
        chat_logdir (str | Path | None):
            If provided, parent folder under which each function_name
            will have a subdirectory for chat logs.
    """

    def __init__(
        self,
        function_map: Dict[str, str],
        providers: Mapping[str, Callable[[], ChatModelWithName] | ChatModelWithName] = LLM_MODELS,
        chat_logdir: str | Path | None = None,
    ) -> None:
        self._providers = providers  # lazily convert these to actually initialized models
        self._function_map = function_map
        self._total_costs = {function_name: 0.0 for function_name in function_map.keys()}
        self._function_models: dict[str, ChatModelWithName] = {}

        # Optional chat log directory
        self._chat_logdir: Path | None = Path(chat_logdir).expanduser().resolve() if chat_logdir else None
        self._chat_logdirs: dict[str, Path] = {}
        if self._chat_logdir is not None:
            for function_name in function_map.keys():
                self._chat_logdirs[function_name] = self._chat_logdir / function_name

    def log_cost(self, invoke_cost: float, function_name: str) -> None:
        self._total_costs[function_name] += invoke_cost

    # ------------- convenience helpers -----------------
    def set_default(self, name: str) -> None:
        """Switch the default model."""
        if name not in self._providers:
            raise KeyError(f"model {name!r} not found")

    def get(self, function_name: str) -> ChatModelWithName:
        """Return a wrapper for a specific provider by function name."""
        model_name = self._function_map.get(function_name)

        if model_name is None:
            raise KeyError(f"function {function_name!r} not found in function map")
        elif model_name not in self._providers:
            raise KeyError(f"model {model_name!r} not found")

        if function_name not in self._function_models:
            self._function_models[function_name] = self._build_model_instance(model_name, function_name)

        return self._function_models[function_name]

    def get_costs(self) -> dict[str, float]:
        return self._total_costs

    def _build_model_instance(self, model_name: str, function_name: str) -> ChatModelWithName:
        provider = self._providers[model_name]

        if callable(provider):
            chat_model = provider()
        elif isinstance(provider, ChatModelWithName):
            chat_model = ChatModelWithName(provider._model, provider.model_name)
        else:
            raise TypeError(f"Unsupported provider type for {model_name!r}: {type(provider)}")

        if not isinstance(chat_model, ChatModelWithName):
            raise TypeError(f"Provider {model_name!r} did not return ChatModelWithName, got {type(chat_model)}")

        chat_dir = self._chat_logdirs.get(function_name) if self._chat_logdir else None
        chat_model.set_log_fn(self.log_cost, function_name, chat_dir)
        return chat_model


class LLMHarness:
    """
    Multithreaded harness for invoking LLM prompts multiple times in parallel.
    """

    def __init__(self, model_name: str):
        if model_name not in LLM_MODELS:
            raise KeyError(f"Model {model_name!r} not found. Available: {list(LLM_MODELS.keys())}")
        self.model_name = model_name

    def _create_model(self) -> ChatModelWithName:
        """Create a fresh model instance for each invocation."""
        return LLM_MODELS[self.model_name]()

    async def _invoke_once(self, prompt: str, index: int) -> Dict[str, Any]:
        """Invoke the prompt once and return result with metadata."""
        model = self._create_model()
        try:
            result = await model.ainvoke(prompt)
            content = result.content if hasattr(result, "content") else str(result)
            return {
                "index": index,
                "status": "success",
                "content": content,
                "model": self.model_name,
            }
        except Exception as e:
            print(f"[Harness] Invocation {index} failed: {e}")
            return {
                "index": index,
                "status": "error",
                "error": str(e),
                "model": self.model_name,
            }

    async def invoke_parallel(self, prompt: str, n: int) -> List[Dict[str, Any]]:
        """
        Invoke the prompt n times in parallel using asyncio.

        Args:
            prompt: The prompt text to send to the model.
            n: Number of times to invoke.

        Returns:
            List of result dictionaries with index, status, content/error, and model.
        """
        tasks = [self._invoke_once(prompt, i) for i in range(n)]
        results = await asyncio.gather(*tasks)
        return list(results)

    def invoke_parallel_sync(self, prompt: str, n: int) -> List[Dict[str, Any]]:
        """Synchronous wrapper for invoke_parallel."""
        return asyncio.run(self.invoke_parallel(prompt, n))
