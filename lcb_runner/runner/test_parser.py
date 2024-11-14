import unittest
import argparse
from unittest.mock import patch
from lcb_runner.runner.parser import get_args

class TestParser(unittest.TestCase):

    def test_default_values(self):
        with patch('sys.argv', ['script.py']):
            args = get_args()
            self.assertEqual(args.model, "gpt-3.5-turbo-0301")
            self.assertEqual(args.scenario, "codegeneration")
            self.assertEqual(args.n, 10)
            self.assertEqual(args.temperature, 0.2)
            self.assertEqual(args.top_p, 0.95)
            self.assertEqual(args.max_tokens, 2000)
            self.assertEqual(args.stop, ["###"])

    def test_custom_values(self):
        test_args = [
            'script.py',
            '--model', 'custom-model',
            '--scenario', 'selfrepair',
            '--n', '5',
            '--temperature', '0.5',
            '--top_p', '0.8',
            '--max_tokens', '1000',
            '--stop', 'END,STOP'
        ]
        with patch('sys.argv', test_args):
            args = get_args()
            self.assertEqual(args.model, "custom-model")
            self.assertEqual(args.scenario, "selfrepair")
            self.assertEqual(args.n, 5)
            self.assertEqual(args.temperature, 0.5)
            self.assertEqual(args.top_p, 0.8)
            self.assertEqual(args.max_tokens, 1000)
            self.assertEqual(args.stop, ["END", "STOP"])

    def test_boolean_flags(self):
        test_args = [
            'script.py',
            '--trust_remote_code',
            '--not_fast',
            '--cot_code_execution',
            '--continue_existing',
            '--use_cache',
            '--debug',
            '--evaluate'
        ]
        with patch('sys.argv', test_args):
            args = get_args()
            self.assertTrue(args.trust_remote_code)
            self.assertTrue(args.not_fast)
            self.assertTrue(args.cot_code_execution)
            self.assertTrue(args.continue_existing)
            self.assertTrue(args.use_cache)
            self.assertTrue(args.debug)
            self.assertTrue(args.evaluate)

    def test_tensor_parallel_size(self):
        with patch('sys.argv', ['script.py']):
            with patch('torch.cuda.device_count', return_value=4):
                args = get_args()
                self.assertEqual(args.tensor_parallel_size, 4)

        test_args = ['script.py', '--tensor_parallel_size', '2']
        with patch('sys.argv', test_args):
            args = get_args()
            self.assertEqual(args.tensor_parallel_size, 2)

    def test_multiprocess(self):
        with patch('sys.argv', ['script.py']):
            with patch('os.cpu_count', return_value=8):
                args = get_args()
                self.assertEqual(args.multiprocess, 8)

        test_args = ['script.py', '--multiprocess', '4']
        with patch('sys.argv', test_args):
            args = get_args()
            self.assertEqual(args.multiprocess, 4)

if __name__ == '__main__':
    unittest.main()
