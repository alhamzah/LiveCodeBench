import os
from time import sleep

try:
    from anthropic import Anthropic
except ImportError as e:
    pass

from lcb_runner.runner.base_runner import BaseRunner


class ClaudeRunner(BaseRunner):
    """
    Runner class for Claude API interactions.
    Handles completion creation and response processing for Claude models.
    """
    client = Anthropic(api_key=os.getenv("ANTHROPIC_KEY"))

    def __init__(self, args, model):
        """
        Initialize the Claude runner with specified arguments and model.

        Args:
            args: Configuration arguments containing model parameters
            model: The specific Claude model to use
        """
        super().__init__(args, model)
        self.client_kwargs: dict[str | str] = {
            "model": args.model,
            "temperature": args.temperature,
            "max_tokens_to_sample": args.max_tokens,
            "top_p": args.top_p,
        }

    def _run_single(self, prompt: str) -> list[str]:
        """
        Execute a single run of the model with the given prompt.

        Args:
            prompt (str): The input prompt for the model

        Returns:
            list[str]: List of generated outputs

        Raises:
            Exception: If the model execution fails after all retries
        """
        def __run_single(counter):
            try:
                response = self.client.completions.create(
                    prompt=prompt,
                    **self.client_kwargs,
                )                                
                content = response.completion
                return content
            except Exception as e:
                print("Exception: ", repr(e), "Sleeping for 20 seconds...")
                sleep(20 * (11 - counter))
                counter = counter - 1
                if counter == 0:
                    print(f"Failed to run model for {prompt}!")
                    print("Exception: ", repr(e))
                    raise e
                return __run_single(counter)

        outputs = []
        try:
            for _ in range(self.args.n):
                outputs.append(__run_single(10))
        except Exception as e:
            raise e

        return outputs