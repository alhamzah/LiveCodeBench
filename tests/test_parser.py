import unittest
import argparse
from unittest.mock import patch
from lcb_runner.runner.parser import get_args
from lcb_runner.utils.scenarios import Scenario

class TestParser(unittest.TestCase):
    def test_default_arguments(self):
        with patch('argparse.ArgumentParser.parse_args',
                   return_value=argparse.Namespace()):
            args = get_args()
            self.assertEqual(args.model, "gpt-3.5-turbo-0301")
            self.assertEqual(args.local_model_path, None)
            self.assertFalse(args.trust_remote_code)
            self.assertEqual(args.scenario, Scenario.codegeneration)
            self.assertFalse(args.not_fast)
            self.assertEqual(args.release_version, "release_v1")
            self.assertFalse(args.cot_code_execution)
            self.assertEqual(args.n, 10)
            self.assertEqual(args.codegen_n, 10)
            self.assertEqual(args.temperature, 0.2)
            self.assertEqual(args.top_p, 0.95)
            self.assertEqual(args.max_tokens, 2000)
            self.assertEqual(args.multiprocess, 0)
            self.assertEqual(args.stop, ["###"])
            self.assertFalse(args.continue_existing)
            self.assertFalse(args.continue_existing_with_eval)
            self.assertFalse(args.use_cache)
            self.assertEqual(args.cache_batch_size, 100)
            self.assertFalse(args.debug)
            self.assertFalse(args.evaluate)
            self.assertEqual(args.num_process_evaluate, 12)
            self.assertEqual(args.timeout, 6)
            self.assertEqual(args.openai_timeout, 45)
            self.assertEqual(args.tensor_parallel_size, 1)
            self.assertFalse(args.enable_prefix_caching)
            self.assertEqual(args.custom_output_file, None)
            self.assertEqual(args.custom_output_save_name, None)
            self.assertEqual(args.dtype, "bfloat16")

    def test_custom_arguments(self):
        test_args = [
            "--model", "gpt-4",
            "--local_model_path", "/path/to/model",
            "--trust_remote_code",
            "--scenario", "selfrepair",
            "--not_fast",
            "--release_version", "release_v2",
            "--cot_code_execution",
            "--n", "20",
            "--codegen_n", "15",
            "--temperature", "0.5",
            "--top_p", "0.9",
            "--max_tokens", "3000",
            "--multiprocess", "4",
            "--stop", "END,STOP",
            "--continue_existing",
            "--use_cache",
            "--cache_batch_size", "200",
            "--debug",
            "--evaluate",
            "--num_process_evaluate", "8",
            "--timeout", "10",
            "--openai_timeout", "60",
            "--tensor_parallel_size", "2",
            "--enable_prefix_caching",
            "--custom_output_file", "/path/to/output",
            "--custom_output_save_name", "custom_results",
            "--dtype", "float16"
        ]
        
        with patch('sys.argv', ['script_name.py'] + test_args):
            args = get_args()
            self.assertEqual(args.model, "gpt-4")
            self.assertEqual(args.local_model_path, "/path/to/model")
            self.assertTrue(args.trust_remote_code)
            self.assertEqual(args.scenario, Scenario.selfrepair)
            self.assertTrue(args.not_fast)
            self.assertEqual(args.release_version, "release_v2")
            self.assertTrue(args.cot_code_execution)
            self.assertEqual(args.n, 20)
            self.assertEqual(args.codegen_n, 15)
            self.assertEqual(args.temperature, 0.5)
            self.assertEqual(args.top_p, 0.9)
            self.assertEqual(args.max_tokens, 3000)
            self.assertEqual(args.multiprocess, 4)
            self.assertEqual(args.stop, ["END", "STOP"])
            self.assertTrue(args.continue_existing)
            self.assertTrue(args.use_cache)
            self.assertEqual(args.cache_batch_size, 200)
            self.assertTrue(args.debug)
            self.assertTrue(args.evaluate)
            self.assertEqual(args.num_process_evaluate, 8)
            self.assertEqual(args.timeout, 10)
            self.assertEqual(args.openai_timeout, 60)
            self.assertEqual(args.tensor_parallel_size, 2)
            self.assertTrue(args.enable_prefix_caching)
            self.assertEqual(args.custom_output_file, "/path/to/output")
            self.assertEqual(args.custom_output_save_name, "custom_results")
            self.assertEqual(args.dtype, "float16")

if __name__ == '__main__':
    unittest.main()
