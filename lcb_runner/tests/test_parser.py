import unittest
import argparse
from lcb_runner.runner.parser import get_args
from lcb_runner.utils.scenarios import Scenario

class TestParser(unittest.TestCase):
    def test_default_args(self):
        # Test with no arguments
        args = get_args()
        
        self.assertEqual(args.model, "gpt-3.5-turbo-0301")
        self.assertIsNone(args.local_model_path)
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
        self.assertFalse(args.enable_prefix_caching)
        self.assertIsNone(args.custom_output_file)
        self.assertIsNone(args.custom_output_save_name)
        self.assertEqual(args.dtype, "bfloat16")

    def test_custom_args(self):
        test_args = [
            "--model", "gpt-4",
            "--local_model_path", "/path/to/model",
            "--trust_remote_code",
            "--scenario", "codegeneration",
            "--not_fast",
            "--release_version", "v2",
            "--cot_code_execution",
            "--n", "20",
            "--codegen_n", "15",
            "--temperature", "0.8",
            "--top_p", "0.99",
            "--max_tokens", "4000",
            "--multiprocess", "4",
            "--stop", "END,STOP",
            "--continue_existing",
            "--use_cache",
            "--cache_batch_size", "200",
            "--debug",
            "--evaluate",
            "--num_process_evaluate", "24",
            "--timeout", "10",
            "--openai_timeout", "60",
            "--tensor_parallel_size", "2",
            "--enable_prefix_caching",
            "--custom_output_file", "output.json",
            "--custom_output_save_name", "test_run",
            "--dtype", "float16"
        ]

        args = get_args()
        args = argparse.Namespace(**vars(args))  # Create copy
        for i in range(0, len(test_args), 2):
            if len(test_args) > i+1:
                setattr(args, test_args[i][2:], test_args[i+1])
            else:
                setattr(args, test_args[i][2:], True)

        self.assertEqual(args.model, "gpt-4")
        self.assertEqual(args.local_model_path, "/path/to/model")
        self.assertTrue(args.trust_remote_code)
        self.assertEqual(args.scenario, Scenario.codegeneration)
        self.assertTrue(args.not_fast)
        self.assertEqual(args.release_version, "v2")
        self.assertTrue(args.cot_code_execution)
        self.assertEqual(args.n, "20")
        self.assertEqual(args.codegen_n, "15")
        self.assertEqual(args.temperature, "0.8")
        self.assertEqual(args.top_p, "0.99")
        self.assertEqual(args.max_tokens, "4000")
        self.assertEqual(args.multiprocess, "4")
        self.assertEqual(args.stop, ["END", "STOP"])
        self.assertTrue(args.continue_existing)
        self.assertTrue(args.use_cache)
        self.assertEqual(args.cache_batch_size, "200")
        self.assertTrue(args.debug)
        self.assertTrue(args.evaluate)
        self.assertEqual(args.num_process_evaluate, "24")
        self.assertEqual(args.timeout, "10")
        self.assertEqual(args.openai_timeout, "60")
        self.assertTrue(args.enable_prefix_caching)
        self.assertEqual(args.custom_output_file, "output.json")
        self.assertEqual(args.custom_output_save_name, "test_run")
        self.assertEqual(args.dtype, "float16")

if __name__ == '__main__':
    unittest.main()
