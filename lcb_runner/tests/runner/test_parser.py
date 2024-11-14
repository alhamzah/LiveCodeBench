# Add imports
import pytest
import argparse
from lcb_runner.runner.parser import get_args
from lcb_runner.utils.scenarios import Scenario

# Add test functions:
def test_default_args():
    args = get_args()
    # Test default values
    assert args.model == "gpt-3.5-turbo-0301"
    assert args.scenario == Scenario.codegeneration
    assert args.n == 10
    assert args.temperature == 0.2

def test_custom_args(monkeypatch):
    test_args = ["--model", "gpt-4", "--scenario", "codeexecution", "--n", "5"]
    monkeypatch.setattr('sys.argv', ['script.py'] + test_args)
    args = get_args()
    assert args.model == "gpt-4"
    assert args.scenario == Scenario.codeexecution
    assert args.n == 5

def test_invalid_scenario(monkeypatch):
    test_args = ["--scenario", "invalid_scenario"]
    monkeypatch.setattr('sys.argv', ['script.py'] + test_args)
    with pytest.raises(ValueError):
        get_args()
