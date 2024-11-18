import unittest
from unittest.mock import patch
from lcb_runner.utils.path_utils import ensure_dir, get_cache_path, get_output_path, get_eval_all_output_path
from lcb_runner.utils.scenarios import Scenario

class TestPathUtils(unittest.TestCase):

    @patch('pathlib.Path.mkdir')
    def test_ensure_dir(self, mock_mkdir):
        ensure_dir('/test/path/file.txt')
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

        ensure_dir('/test/path/', is_file=False)
        mock_mkdir.assert_called_with(parents=True, exist_ok=True)

    def test_get_cache_path(self):
        args = unittest.mock.Mock()
        args.scenario = Scenario.PYTHON
        args.n = 5
        args.temperature = 0.7
        
        expected_path = "cache/test_model/PYTHON_5_0.7.json"
        self.assertEqual(get_cache_path("test_model", args), expected_path)

    def test_get_output_path(self):
        args = unittest.mock.Mock()
        args.scenario = Scenario.PYTHON
        args.n = 5
        args.temperature = 0.7
        args.cot_code_execution = True
        
        expected_path = "output/test_model/PYTHON_5_0.7_cot.json"
        self.assertEqual(get_output_path("test_model", args), expected_path)

    def test_get_eval_all_output_path(self):
        args = unittest.mock.Mock()
        args.scenario = Scenario.PYTHON
        args.n = 5
        args.temperature = 0.7
        args.cot_code_execution = False
        
        expected_path = "output/test_model/PYTHON_5_0.7_eval_all.json"
        self.assertEqual(get_eval_all_output_path("test_model", args), expected_path)

if __name__ == '__main__':
    unittest.main()
