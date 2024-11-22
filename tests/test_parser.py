import unittest
import argparse
from lcb_runner.runner.parser import get_args
from lcb_runner.utils.scenarios import Scenario

class TestParser(unittest.TestCase):
    def test_default_args(self):
        args = get_args()
        self.assertEqual(args.model, "gpt-3.5-turbo-0301")
        self.assertEqual(args.scenario, Scenario.codegeneration)
        self.assertEqual(args.n, 10)
        self.assertEqual(args.temperature, 0.2)
        self.assertEqual(args.top_p, 0.95)
        self.assertEqual(args.max_tokens, 2000)

    def test_custom_args(self):
        test_args = [
            "--model", "gpt-4",
            "--scenario", "codeexecution",
            "--n", "20",
            "--temperature", "0.5",
            "--top_p", "0.9",
            "--max_tokens", "1000"
        ]
        args = get_args(test_args)
        self.assertEqual(args.model, "gpt-4")
        self.assertEqual(args.scenario, Scenario.codeexecution)
        self.assertEqual(args.n, 20)
        self.assertEqual(args.temperature, 0.5)
        self.assertEqual(args.top_p, 0.9)
        self.assertEqual(args.max_tokens, 1000)

    def test_boolean_flags(self):
        test_args = [
            "--not_fast",
            "--cot_code_execution",
            "--continue_existing",
            "--use_cache",
            "--debug",
            "--evaluate"
        ]
        args = get_args(test_args)
        self.assertTrue(args.not_fast)
        self.assertTrue(args.cot_code_execution)
        self.assertTrue(args.continue_existing)
        self.assertTrue(args.use_cache)
        self.assertTrue(args.debug)
        self.assertTrue(args.evaluate)

    def test_stop_token_parsing(self):
        test_args = ["--stop", "###,END"]
        args = get_args(test_args)
        self.assertEqual(args.stop, ["###", "END"])

if __name__ == '__main__':
    unittest.main()
