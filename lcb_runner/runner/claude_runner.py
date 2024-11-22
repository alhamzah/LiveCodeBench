import os
from time import sleep
import logging
from typing import List, Dict, Any
from functools import wraps

try:
    from anthropic import Anthropic
except ImportError as e:
    pass

from lcb_runner.runner.base_runner import BaseRunner


def retry_with_exponential_backoff(
    max_retries: int = 5,
    base_delay: float = 1,
    max_delay: float = 60,
):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            delay = base_delay
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    if retries == max_retries:
                        raise e
                    sleep_time = min(delay * (2 ** retries), max_delay)
                    logging.warning(f"Retry {retries}/{max_retries} after {sleep_time:.2f}s due to {str(e)}")
                    sleep(sleep_time)
            return func(*args, **kwargs)
        return wrapper
    return decorator


class ClaudeRunner(BaseRunner):
    client = Anthropic(api_key=os.getenv("ANTHROPIC_KEY"))

    def __init__(self, args: Any, model: str):
        super().__init__(args, model)
        self.client_kwargs: Dict[str, Any] = {
            "model": args.model,
            "temperature": args.temperature,
            "max_tokens_to_sample": args.max_tokens,
            "top_p": args.top_p,
        }

    @retry_with_exponential_backoff()
    def _run_single(self, prompt: str) -> str:
        logging.info(f"Running Claude model with prompt: {prompt[:50]}...")
        response = self.client.completions.create(
            prompt=prompt,
            **self.client_kwargs,
        )
        content = response.completion
        logging.info(f"Claude model response received: {content[:50]}...")
        return content

    def run(self, prompts: List[str]) -> List[List[str]]:
        outputs = []
        for prompt in prompts:
            prompt_outputs = []
            for _ in range(self.args.n):
                prompt_outputs.append(self._run_single(prompt))
            outputs.append(prompt_outputs)
        return outputs
