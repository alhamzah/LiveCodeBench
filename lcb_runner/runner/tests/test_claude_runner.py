import unittest
from unittest.mock import patch, MagicMock
from lcb_runner.runner.claude_runner import ClaudeRunner

class TestClaudeRunner(unittest.TestCase):
    def setUp(self):
        self.mock_args = MagicMock()
        self.mock_args.model = "claude-2"
        self.mock_args.temperature = 0.7
        self.mock_args.max_tokens = 100
        self.mock_args.top_p = 1.0
        self.mock_args.n = 1

    @patch('lcb_runner.runner.claude_runner.Anthropic')
    def test_run_single(self, mock_anthropic):
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        mock_response = MagicMock()
        mock_response.completion = "Test response"
        mock_client.completions.create.return_value = mock_response

        runner = ClaudeRunner(self.mock_args, "claude-2")
        result = runner._run_single("Test prompt")

        self.assertEqual(result, ["Test response"])
        mock_client.completions.create.assert_called_once_with(
            prompt="Test prompt",
            model="claude-2",
            temperature=0.7,
            max_tokens_to_sample=100,
            top_p=1.0
        )

if __name__ == '__main__':
    unittest.main()

import unittest
from unittest.mock import patch, MagicMock
from lcb_runner.runner.cohere_runner import CohereRunner

class TestCohereRunner(unittest.TestCase):
    def setUp(self):
        self.mock_args = MagicMock()
        self.mock_args.model = "command"
        self.mock_args.temperature = 0.7
        self.mock_args.max_tokens = 100
        self.mock_args.top_p = 1.0
        self.mock_args.n = 1

    @patch('lcb_runner.runner.cohere_runner.cohere')
    def test_run_single(self, mock_cohere):
        mock_client = MagicMock()
        mock_cohere.Client.return_value = mock_client
        mock_response = MagicMock()
        mock_response.text = "Test response"
        mock_client.chat.return_value = mock_response

        runner = CohereRunner(self.mock_args, "command")
        chat_history = [{"role": "user", "message": "Hello"}]
        message = "How are you?"
        result = runner._run_single((chat_history, message))

        self.assertEqual(result, ["Test response"])
        mock_client.chat.assert_called_once_with(
            message="How are you?",
            chat_history=chat_history,
            model="command",
            temperature=0.7,
            max_tokens=100,
            p=1.0
        )

if __name__ == '__main__':
    unittest.main()

class ClaudeRunner(BaseRunner):
    """
    A runner class for the Claude AI model from Anthropic.
    
    This class handles the interaction with the Anthropic API to run completions
    using the Claude model. It includes error handling and retry logic.
    """

class CohereRunner(BaseRunner):
    """
    A runner class for the Cohere AI model.
    
    This class manages the interaction with the Cohere API to run chat completions.
    It includes error handling and retry logic.
    """
