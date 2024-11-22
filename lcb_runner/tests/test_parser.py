import pytest
from lcb_runner.runner.parser import get_args

def test_default_values():
    args = get_args()
    assert args.model == "gpt-3.5-turbo-0301"
    assert args.scenario == "codegeneration"
    assert args.n == 10
    assert args.temperature == 0.2
    # Add more assertions for other default values

def test_custom_values():
    test_args = ["--model", "gpt2", "--scenario", "selfrepair", "--n", "5"]
    args = get_args(test_args)
    assert args.model == "gpt2"
    assert args.scenario == "selfrepair"
    assert args.n == 5

def test_stop_tokens():
    test_args = ["--stop", "a,b,c"]
    args = get_args(test_args)
    assert args.stop == ["a", "b", "c"]

def test_tensor_parallel_size():
    test_args = ["--tensor_parallel_size", "2"]
    args = get_args(test_args)
    assert args.tensor_parallel_size == 2

def test_multiprocess():
    test_args = ["--multiprocess", "4"]
    args = get_args(test_args)
    assert args.multiprocess == 4

# Add more test cases as needed
