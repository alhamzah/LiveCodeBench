import pytest
from lcb_runner.runner.parser import get_args

def test_default_values():
    args = get_args()
    assert args.model == "gpt-3.5-turbo-0301"
    assert args.scenario == "codegeneration"
    # Add more assertions for other default values

def test_custom_values():
    args = get_args(["--model", "gpt3", "--scenario", "codeexecution"])
    assert args.model == "gpt3"
    assert args.scenario == "codeexecution"
    # Add more assertions for other custom values

def test_boolean_flags():
    args = get_args(["--use_cache"])
    assert args.use_cache is True

def test_stop_tokens():
    args = get_args(["--stop", "token1,token2"])
    assert args.stop == ["token1", "token2"]

def test_tensor_parallel_size():
    args = get_args(["--tensor_parallel_size", "4"])
    assert args.tensor_parallel_size == 4

def test_multiprocess():
    args = get_args(["--multiprocess", "2"])
    assert args.multiprocess == 2
