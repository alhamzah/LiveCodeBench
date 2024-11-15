import os
import pathlib
from unittest.mock import Mock

import pytest

from lcb_runner.utils.path_utils import (
    ensure_dir,
    get_cache_path,
    get_output_path,
    get_eval_all_output_path
)


def test_ensure_dir_file(tmp_path):
    test_file = tmp_path / "test_dir" / "test_file.txt"
    ensure_dir(str(test_file), is_file=True)
    assert test_file.parent.exists()
    assert not test_file.exists()


def test_ensure_dir_directory(tmp_path):
    test_dir = tmp_path / "test_dir"
    ensure_dir(str(test_dir), is_file=False)
    assert test_dir.exists()


def test_get_cache_path():
    args = Mock(scenario="test_scenario", n=5, temperature=0.7)
    path = get_cache_path("test_model", args)
    assert path == "cache/test_model/test_scenario_5_0.7.json"


def test_get_output_path_without_cot():
    args = Mock(scenario="test_scenario", n=5, temperature=0.7, cot_code_execution=False)
    path = get_output_path("test_model", args)
    assert path == "output/test_model/test_scenario_5_0.7.json"


def test_get_output_path_with_cot():
    args = Mock(scenario="test_scenario", n=5, temperature=0.7, cot_code_execution=True)
    path = get_output_path("test_model", args)
    assert path == "output/test_model/test_scenario_5_0.7_cot.json"


def test_get_eval_all_output_path_without_cot():
    args = Mock(scenario="test_scenario", n=5, temperature=0.7, cot_code_execution=False)
    path = get_eval_all_output_path("test_model", args)
    assert path == "output/test_model/test_scenario_5_0.7_eval_all.json"


def test_get_eval_all_output_path_with_cot():
    args = Mock(scenario="test_scenario", n=5, temperature=0.7, cot_code_execution=True)
    path = get_eval_all_output_path("test_model", args)
    assert path == "output/test_model/test_scenario_5_0.7_cot_eval_all.json"
