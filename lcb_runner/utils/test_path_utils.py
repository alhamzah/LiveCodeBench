import unittest
import os
from unittest.mock import MagicMock
from lcb_runner.utils.path_utils import ensure_dir, get_cache_path, get_output_path, get_eval_all_output_path

class TestPathUtils(unittest.TestCase):

    def setUp(self):
        # Setup a mock for args
        self.args = MagicMock()
        self.args.scenario = "test_scenario"
        self.args.n = 1
        self.args.temperature = 0.5
        self.args.cot_code_execution = False

    def test_ensure_dir_creates_directory_for_file(self):
        path = "test_dir/test_file.txt"
        ensure_dir(path, is_file=True)
        self.assertTrue(os.path.exists("test_dir"))
        os.rmdir("test_dir")

    def test_ensure_dir_creates_directory(self):
        path = "test_dir"
        ensure_dir(path, is_file=False)
        self.assertTrue(os.path.exists("test_dir"))
        os.rmdir("test_dir")

    def test_get_cache_path(self):
        model_repr = "test_model"
        expected_path = "cache/test_model/test_scenario_1_0.5.json"
        actual_path = get_cache_path(model_repr, self.args)
        self.assertEqual(expected_path, actual_path)

    def test_get_output_path(self):
        model_repr = "test_model"
        expected_path = "output/test_model/test_scenario_1_0.5.json"
        actual_path = get_output_path(model_repr, self.args)
        self.assertEqual(expected_path, actual_path)

    def test_get_eval_all_output_path(self):
        model_repr = "test_model"
        expected_path = "output/test_model/test_scenario_1_0.5_eval_all.json"
        actual_path = get_eval_all_output_path(model_repr, self.args)
        self.assertEqual(expected_path, actual_path)

if __name__ == '__main__':
    unittest.main()
