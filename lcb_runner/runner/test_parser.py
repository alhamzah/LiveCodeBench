import unittest
import argparse
from unittest.mock import patch
import torch
from lcb_runner.runner.parser import get_args
from lcb_runner.utils.scenarios import Scenario

class TestParser(unittest.TestCase):
    def test_default_arguments(self):
        with patch('sys.argv', ['script.py']):
            args = get_args()
            self.assertEqual(args.model, "gpt-3.5-turbo-0301")
            self.assertEqual(args.scenario, Scenario.codegeneration)
            self.assertEqual(args.n, 10)
            self.assertEqual(args.temperature, 0.2)
            self.assertEqual(args.top_p, 0.95)
            self.assertEqual(args.max_tokens, 2000)
            self.assertEqual(args.stop, ["###"])

    def test_custom_arguments(self):
        test_args = [
            'script.py',
            '--model', 'custom-model',
            '--scenario', 'selfrepair',
            '--n', '20',
            '--temperature', '0.5',
            '--top_p', '0.9',
            '--max_tokens', '1000',
            '--stop', 'END,STOP'
        ]
        with patch('sys.argv', test_args):
            args = get_args()
            self.assertEqual(args.model, "custom-model")
            self.assertEqual(args.scenario, Scenario.selfrepair)
            self.assertEqual(args.n, 20)
            self.assertEqual(args.temperature, 0.5)
            self.assertEqual(args.top_p, 0.9)
            self.assertEqual(args.max_tokens, 1000)
            self.assertEqual(args.stop, ["END", "STOP"])

    def test_tensor_parallel_size(self):
        with patch('sys.argv', ['script.py', '--tensor_parallel_size', '4']):
            args = get_args()
            self.assertEqual(args.tensor_parallel_size, 4)

        with patch('sys.argv', ['script.py']):
            with patch('torch.cuda.device_count', return_value=8):
                args = get_args()
                self.assertEqual(args.tensor_parallel_size, 8)

    def test_multiprocess_default(self):
        with patch('sys.argv', ['script.py']):
            args = get_args()
            self.assertEqual(args.multiprocess, 0)

    def test_multiprocess_auto(self):
        with patch('sys.argv', ['script.py', '--multiprocess', '-1']):
            with patch('os.cpu_count', return_value=16):
                args = get_args()
                self.assertEqual(args.multiprocess, 16)

    def test_invalid_arguments(self):
        with self.assertRaises(SystemExit):
            with patch('sys.argv', ['script.py', '--invalid_arg']):
                get_args()

    def test_scenario_enum(self):
        with patch('sys.argv', ['script.py', '--scenario', 'codegeneration']):
            args = get_args()
            self.assertEqual(args.scenario, Scenario.codegeneration)

        with patch('sys.argv', ['script.py', '--scenario', 'selfrepair']):
            args = get_args()
            self.assertEqual(args.scenario, Scenario.selfrepair)

if __name__ == '__main__':
    unittest.main()
