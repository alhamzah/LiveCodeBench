import unittest
from unittest.mock import patch, MagicMock
from lcb_runner.runner.runner_utils import build_runner
from lcb_runner.lm_styles import LMStyle, LanguageModel

class TestBuildRunner(unittest.TestCase):

    @patch('lcb_runner.runner.oai_runner.OpenAIRunner')
    def test_openai_runner(self, MockOpenAIRunner):
        args = MagicMock()
        model = MagicMock(model_style=LMStyle.OpenAIChat)
        runner = build_runner(args, model)
        MockOpenAIRunner.assert_called_once_with(args, model)
        self.assertEqual(runner, MockOpenAIRunner.return_value)

    @patch('lcb_runner.runner.gemini_runner.GeminiRunner')
    def test_gemini_runner(self, MockGeminiRunner):
        args = MagicMock()
        model = MagicMock(model_style=LMStyle.Gemini)
        runner = build_runner(args, model)
        MockGeminiRunner.assert_called_once_with(args, model)
        self.assertEqual(runner, MockGeminiRunner.return_value)

    @patch('lcb_runner.runner.claude3_runner.Claude3Runner')
    def test_claude3_runner(self, MockClaude3Runner):
        args = MagicMock()
        model = MagicMock(model_style=LMStyle.Claude3)
        runner = build_runner(args, model)
        MockClaude3Runner.assert_called_once_with(args, model)
        self.assertEqual(runner, MockClaude3Runner.return_value)

    @patch('lcb_runner.runner.claude_runner.ClaudeRunner')
    def test_claude_runner(self, MockClaudeRunner):
        args = MagicMock()
        model = MagicMock(model_style=LMStyle.Claude)
        runner = build_runner(args, model)
        MockClaudeRunner.assert_called_once_with(args, model)
        self.assertEqual(runner, MockClaudeRunner.return_value)

    @patch('lcb_runner.runner.mistral_runner.MistralRunner')
    def test_mistral_runner(self, MockMistralRunner):
        args = MagicMock()
        model = MagicMock(model_style=LMStyle.MistralWeb)
        runner = build_runner(args, model)
        MockMistralRunner.assert_called_once_with(args, model)
        self.assertEqual(runner, MockMistralRunner.return_value)

    @patch('lcb_runner.runner.cohere_runner.CohereRunner')
    def test_cohere_runner(self, MockCohereRunner):
        args = MagicMock()
        model = MagicMock(model_style=LMStyle.CohereCommand)
        runner = build_runner(args, model)
        MockCohereRunner.assert_called_once_with(args, model)
        self.assertEqual(runner, MockCohereRunner.return_value)

    @patch('lcb_runner.runner.deepseek_runner.DeepSeekRunner')
    def test_deepseek_runner(self, MockDeepSeekRunner):
        args = MagicMock()
        model = MagicMock(model_style=LMStyle.DeepSeekAPI)
        runner = build_runner(args, model)
        MockDeepSeekRunner.assert_called_once_with(args, model)
        self.assertEqual(runner, MockDeepSeekRunner.return_value)

    def test_not_implemented_error(self):
        args = MagicMock()
        model = MagicMock(model_style=None)  # Use a style not in the list
        with self.assertRaises(NotImplementedError):
            build_runner(args, model)

if __name__ == '__main__':
    unittest.main()
