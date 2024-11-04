import os
from time import sleep

try:
    from anthropic import Anthropic
except ImportError as e:
    pass

from lcb_runner.runner.base_runner import BaseRunner


class ClaudeRunner(BaseRunner):
    client = Anthropic(api_key=os.getenv("ANTHROPIC_KEY"))

    def __init__(self, args, model):
        """
        Initialize the ClaudeRunner.

        Args:
            args: Command-line arguments.
            model: The language model to use.
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
        Run the Claude model for a single prompt.

        Args:
            prompt (str): The input prompt.

        Returns:
            list[str]: A list of generated outputs.
        """

        def __run_single(counter):
            """
            Helper function to run a single prompt with retry logic.

            Args:
                counter (int): Number of retries left.

            Returns:
                str: The generated content.

            Raises:
                Exception: If all retries fail.
            """
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
