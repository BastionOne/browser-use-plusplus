from __future__ import annotations  # type: ignore[all]
import time
import asyncio
from pathlib import Path
from datetime import datetime

from jinja2 import Template, Environment, meta
import json

from logging import Logger
from collections.abc import Iterable
from typing import Callable, Dict, Generic, Any, TypeVar, get_args, get_origin, List, Optional, Type, Tuple, Set
import opik

from pydantic import create_model

from instructor.dsl.iterable import IterableModel
from instructor.dsl.simple_type import ModelAdapter, is_simple_type
from instructor.function_calls import OpenAISchema, openai_schema

from bupp.src.llm.llm_models import openai_41 as lazy_openai_41

T = TypeVar("T")


def log_prompt_to_files(
    prompt_logdir: Path | None,
    prompt_template: str,
    prompt_args: Dict[str, Any],
    lmp_name: str = ""
) -> None:
    """
    Logs the prompt template and prompt_args to separate files.
    
    Args:
        prompt_logdir: Directory to write log files to
        prompt_template: The raw prompt template string (without variables substituted)
        prompt_args: The arguments that were passed to render the template
        lmp_name: Name of the LMP class for the file prefix
    """
    if prompt_logdir is None:
        return

    prompt_logdir.mkdir(parents=True, exist_ok=True)

    # Find the next available log file number
    counter = 1
    while True:
        template_file = prompt_logdir / f"{counter}_template.txt"
        if not template_file.exists():
            break
        counter += 1

    # Write prompt template
    template_file = prompt_logdir / f"{counter}_template.txt"
    with open(template_file, "w", encoding="utf-8") as f:
        if lmp_name:
            f.write(f"# LMP: {lmp_name}\n")
            f.write(f"# Timestamp: {datetime.now().isoformat()}\n\n")
        f.write(prompt_template)

    # Write prompt_args as JSON
    args_file = prompt_logdir / f"{counter}_args.json"
    with open(args_file, "w", encoding="utf-8") as f:
        # Convert non-JSON-serializable values to strings
        serializable_args = {}
        for k, v in prompt_args.items():
            try:
                json.dumps(v)
                serializable_args[k] = v
            except (TypeError, ValueError):
                serializable_args[k] = str(v)
        json.dump(serializable_args, f, indent=2, ensure_ascii=False) 

def extract_json(response: str) -> str:
    """
    Extracts the JSON from the response using stack-based parsing to match braces.
    """
    # First try to extract from markdown code blocks
    try:
        if "```json" in response:
            return response.split("```json")[1].split("```")[0]
    except IndexError:
        pass
    
    # Find the first opening brace
    start_idx = response.find("{")
    if start_idx == -1:
        # No JSON found, return original response
        return response
    
    # Use stack-based parsing to find matching closing brace
    stack = []
    for i, char in enumerate(response[start_idx:], start_idx):
        if char == "{":
            stack.append(char)
        elif char == "}":
            if stack:
                stack.pop()
                if not stack:
                    # Found matching closing brace
                    return response[start_idx:i+1]
    
    # If we get here, unmatched braces - return from start to end
    return response[start_idx:]

def is_typed_dict(cls) -> bool:
    return (
        isinstance(cls, type)
        and issubclass(cls, dict)
        and hasattr(cls, "__annotations__")
    )

# pylint: skip-file
def prepare_response_model(response_model: type[Any] | None) -> Any:
    """
    Prepares the response model for use in the API call.

    This function performs several transformations on the input response_model:
    1. If the response_model is None, it returns None.
    2. If it's a simple type, it wraps it in a ModelAdapter.
    3. If it's a TypedDict, it converts it to a Pydantic BaseModel.
    4. If it's an Iterable, it wraps the element type in an IterableModel.
    5. If it's not already a subclass of OpenAISchema, it applies the openai_schema decorator.

    Args:
        response_model (type[T] | None): The input response model to be prepared.

    Returns:
        Any: The prepared response model, or None if the input was None.
    """
    if response_model is None:
        return None

    if is_simple_type(response_model):
        response_model = ModelAdapter[response_model]

    if is_typed_dict(response_model):
        response_model = create_model(  # type: ignore
            response_model.__name__,
            **{k: (v, ...) for k, v in response_model.__annotations__.items()},  # type: ignore
        )

    if get_origin(response_model) is Iterable:
        iterable_element_class = get_args(response_model)[0]
        response_model = IterableModel(iterable_element_class)

    if not issubclass(response_model, OpenAISchema):  # type: ignore
        response_model = openai_schema(response_model)  # type: ignore

    return response_model

def get_instructor_prompt(response_format: Any) -> str:
    if not response_format:
        return ""

    response_model = prepare_response_model(response_format)
    return f"""
    \n
Understand the content and provide the parsed objects in json that match the following json_schema:\n

{json.dumps(response_model.model_json_schema(), indent=2, ensure_ascii=False)}

Make sure to return an instance of the JSON, not the schema itself
    """

def get_all_template_variables(template_str: str) -> Set[str]:
    """
    Extract all variable references in a Jinja2 template.
    
    Returns:
        Set of all variable names used in template
    """
    env = Environment()
    ast = env.parse(template_str)
    
    return meta.find_undeclared_variables(ast)

class LMPVerificationException(Exception):
    """Thrown when post_process raises an error"""
    pass

# TODO: change to use generic[t]
# DESIGN: not sure how to enforce this but we should only allow JSON serializable
# args to be passed to the model, to be compatible with Braintrust 
class LMP(Generic[T]):
    """
    A language model progsram
    """
    prompt: str
    response_format: Any
    templates: Dict = {}
    opik_prompt: Optional[opik.Prompt] = None
    prompt_logdir: Optional[Path] = None

    def __init__(self, opik_config: Optional[Dict] = None):
        self._error_message = None

        if opik_config:
        # we reassign the prompt template according to opik_config
            prompt_name, commit = opik_config["name"], opik_config.get("commit", None)
            opik_client = opik.Opik()
            opik_prompt = opik_client.get_prompt(prompt_name, commit=commit)
            if not opik_prompt:
                raise ValueError(f"Prompt {prompt_name} not found")

            self.prompt = opik_prompt.prompt
            self.opik_prompt = opik_prompt

        elif not self.prompt:
            raise ValueError("Either prompt cls var need to be declared or opik_config must be provided")

    def _prepare_prompt(
        self, 
        templates={}, 
        **prompt_args
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Prepares the prompt string by rendering the template with provided arguments.
        
        Returns:
            Tuple of (rendered_prompt_string, combined_prompt_args)
        """
        template_vars = get_all_template_variables(self.prompt)
        if self._error_message:
            if "error_message" in template_vars:
                prompt_args["error_message"] = self._error_message
            else:
                raise ValueError(f"Error message provided but 'error_message' is not a template variable in the prompt")

        prompt_str = self.prompt + self._get_instructor_prompt()
        
        # Combine prompt_args and templates for logging
        combined_args = {**prompt_args, **templates}

        prompt_str = Template(prompt_str).render(**prompt_args, **templates)
        return prompt_str, combined_args

    def set_error_message(self, error_message: str) -> None:
        self._error_message = error_message

    def _verify_or_raise(self, res, **prompt_args):
        return True

    def _process_result(self, res, **prompt_args) -> Any:
        return res
    
    def _get_instructor_prompt(self) -> str:
        if not self.response_format:
            return ""
    
        response_model = prepare_response_model(self.response_format)
        return f"""
        \n\n
Understand the content and provide the parsed objects in json that match the following json_schema:\n

{json.dumps(response_model.model_json_schema(), indent=2, ensure_ascii=False)}

Make sure to return an instance of the JSON, not the schema itself

        """
    
    def invoke_with_msgs(
            self, 
            model: Any,
            msgs: List[Any],
            **prompt_args
        ) -> Any:
        res = model.invoke(msgs)
        content = res.content   

        if not isinstance(content, str):
            raise Exception("[LLM] CONTENT IS NOT A STRING")
        
        if self.response_format:
            content = extract_json(content)
            content = self.response_format.model_validate_json(content)

        self._verify_or_raise(content, **prompt_args)
        # skip process_result
        return content

    def invoke(self, 
            model: Any,
            max_retries: int = 5,
            retry_delay: int = 1,
            prompt_args: Dict = {},
            log_this_prompt: Optional[Logger] = None,
            prompt_log_preamble: Optional[str] = ""
        ) -> Any:
        prompt, combined_args = self._prepare_prompt(
            templates=self.templates,
            **prompt_args,
        )
        
        # Log prompt template and args to files
        log_prompt_to_files(
            self.prompt_logdir,
            self.prompt,  # raw template without variables
            combined_args,
            self.__class__.__name__
        )
        
        if log_this_prompt:
            log_this_prompt.info(f"{prompt_log_preamble}\n[{self.__class__.__name__}]: {prompt}")

        # TODO: make this decorator
        current_retry = 1
        while current_retry <= max_retries:
            try:
                res = model.invoke(prompt)
                content = res.content

                if not isinstance(content, str):
                    raise Exception("[LLM] CONTENT IS NOT A STRING")
                
                if self.response_format:
                    try:
                        content = extract_json(content)
                        content = self.response_format.model_validate_json(content)
                    except Exception as e:
                        print(f"Error validating response: {e}")
                        print(f"Response:\n -------------\n{content}\n -------------")
                        raise e
        
                self._verify_or_raise(content, **prompt_args)
                return self._process_result(content, **prompt_args)
            
            except Exception as e:
                current_retry += 1
                
                if current_retry > max_retries:
                    raise e
                
                # Exponential backoff: retry_delay * (2 ^ attempt)
                current_delay = retry_delay * (2 ** (current_retry - 1))
                time.sleep(current_delay)
                print(f"Retry attempt {current_retry}/{max_retries} after error: {str(e)}. Waiting {current_delay}s")

    async def ainvoke(
        self,
        model: Any,
        max_retries: int = 3,
        retry_delay: int = 1,
        prompt_args: Dict = {},
        prompt_logger: Optional[Logger] = None,
        prompt_log_preamble: Optional[str] = "",
        dry_run: bool = False,
        clean_res: Callable[str,[str]] = None
    ) -> Any:
        """Async version of invoke that leverages model.ainvoke when available.

        Falls back to running the sync invoke in a thread if the model has no ainvoke.
        """
        prompt, combined_args = self._prepare_prompt(
            templates=self.templates,
            **prompt_args,
        )
        
        # Log prompt template and args to files
        log_prompt_to_files(
            self.prompt_logdir,
            self.prompt,  # raw template without variables
            combined_args,
            self.__class__.__name__
        )
        
        if prompt_logger:
            prompt_logger.info(f"{prompt_log_preamble}\n[{self.__class__.__name__}]: {prompt}")
        
        if dry_run:
            print(prompt)
            return

        current_retry = 1
        while current_retry <= max_retries:
            try:
                # Invoke primary model
                if hasattr(model, "ainvoke"):
                    res = await model.ainvoke(prompt)
                    # Manually log cost since wrapper logging happens in sync path
                    if hasattr(model, "log_cost"):
                        try:
                            model.log_cost(res)
                        except Exception:
                            pass
                else:
                    res = await asyncio.to_thread(model.invoke, prompt)

                content = res.completion

                if not isinstance(content, str):
                    raise Exception("[LLM] CONTENT IS NOT A STRING")

                if self.response_format:
                    try:
                        if clean_res:
                            content = clean_res(content)
                        content = extract_json(content)
                        content = self.response_format.model_validate_json(content)
                    except Exception as e:
                        print(f"Error validating response: {e}")
                        print(f"Response:\n -------------\n{content}\n -------------")
                        raise e

                self._verify_or_raise(content, **prompt_args)
                return self._process_result(content, **prompt_args)

            except Exception as e:
                current_retry += 1
                if current_retry > max_retries:
                    raise e
                current_delay = retry_delay * (2 ** (current_retry - 1))
                await asyncio.sleep(current_delay)
                print(f"Retry attempt {current_retry}/{max_retries} after error: {str(e)}. Waiting {current_delay}s")

    def get_opik_prompt_info(self) -> Tuple[str, str]:
        if self.opik_prompt:
            return self.opik_prompt.name, self.opik_prompt.commit
        raise ValueError("No opik prompt found")