import unittest
from unittest.mock import patch
from lcb_runner.runner.parser import get_args

class TestParser(unittest.TestCase):

    @patch('argparse._sys.argv', ['parser.py', '--model', 'gpt-4', '--n', '5'])
    def test_get_args_default(self):
        args = get_args()
        self.assertEqual(args.model, 'gpt-4')
        self.assertEqual(args.n, 5)

    @patch('argparse._sys.argv', ['parser.py', '--temperature', '0.5', '--top_p', '0.9'])
    def test_get_args_sampling(self):
        args = get_args()
        self.assertEqual(args.temperature, 0.5)
        self.assertEqual(args.top_p, 0.9)

    @patch('argparse._sys.argv', ['parser.py', '--enable_prefix_caching'])
    def test_get_args_flags(self):
        args = get_args()
        self.assertTrue(args.enable_prefix_caching)

if __name__ == '__main__':
    unittest.main()
