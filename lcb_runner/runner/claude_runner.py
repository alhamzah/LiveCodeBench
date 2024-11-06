import os
from time import sleep

try:
    from anthropic import Anthropic
except ImportError as e:
    pass

from lcb_runner.runner.base_runner import BaseRunner


class ClaudeRunner(BaseRunner):
    """Runner class for the Claude model from Anthropic.
    
    This class handles running inference using the Claude API, with retry logic
    and proper error handling. It inherits from BaseRunner and implements the
    model-specific logic for Claude.
    """
    
    client = Anthropic(api_key=os.getenv("ANTHROPIC_KEY"))

    def __init__(self, args, model):
        """Initialize the Claude runner.
        
        Args:
            args: Arguments containing model configuration like temperature, max tokens etc.
            model: The specific Claude model to use (e.g. claude-2)
        """
        super().__init__(args, model)
        self.client_kwargs: dict[str | str] = {
            "model": args.model,
            "temperature": args.temperature,
            "max_tokens_to_sample": args.max_tokens,
            "top_p": args.top_p,
        }

    def _run_single(self, prompt: str) -> list[str]:
        """Run a single inference pass with the Claude model.
        
        Args:
            prompt: The input prompt to send to Claude
            
        Returns:
            list[str]: List of model outputs for the given prompt
            
        Raises:
            Exception: If all retry attempts fail
        """

        def __run_single(counter):
            """Helper function that handles a single API call with retries.
            
            Args:
                counter: Number of remaining retry attempts
                
            Returns:
                str: The model's response text
                
            Raises:
                Exception: If all retry attempts are exhausted
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
