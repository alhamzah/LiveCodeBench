import unittest
from unittest.mock import patch
import pathlib
from lcb_runner.utils.path_utils import ensure_dir, get_cache_path, get_output_path, get_eval_all_output_path
from lcb_runner.utils.scenarios import Scenario

class TestPathUtils(unittest.TestCase):

    @patch('pathlib.Path.mkdir')
    def test_ensure_dir_file(self, mock_mkdir):
        ensure_dir('/path/to/file.txt')
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    @patch('pathlib.Path.mkdir')
    def test_ensure_dir_directory(self, mock_mkdir):
        ensure_dir('/path/to/directory', is_file=False)
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    @patch('lcb_runner.utils.path_utils.ensure_dir')
    def test_get_cache_path(self, mock_ensure_dir):
        args = unittest.mock.Mock()
        args.scenario = Scenario.PYTHON
        args.n = 5
        args.temperature = 0.7
        
        result = get_cache_path('gpt-3.5-turbo', args)
        
        expected_path = 'cache/gpt-3.5-turbo/PYTHON_5_0.7.json'
        self.assertEqual(result, expected_path)
        mock_ensure_dir.assert_called_once_with(expected_path)

    @patch('lcb_runner.utils.path_utils.ensure_dir')
    def test_get_output_path(self, mock_ensure_dir):
        args = unittest.mock.Mock()
        args.scenario = Scenario.PYTHON
        args.n = 5
        args.temperature = 0.7
        args.cot_code_execution = True
        
        result = get_output_path('gpt-3.5-turbo', args)
        
        expected_path = 'output/gpt-3.5-turbo/PYTHON_5_0.7_cot.json'
        self.assertEqual(result, expected_path)
        mock_ensure_dir.assert_called_once_with(expected_path)

    def test_get_eval_all_output_path(self):
        args = unittest.mock.Mock()
        args.scenario = Scenario.PYTHON
        args.n = 5
        args.temperature = 0.7
        args.cot_code_execution = False
        
        result = get_eval_all_output_path('gpt-3.5-turbo', args)
        
        expected_path = 'output/gpt-3.5-turbo/PYTHON_5_0.7_eval_all.json'
        self.assertEqual(result, expected_path)

if __name__ == '__main__':
    unittest.main()
