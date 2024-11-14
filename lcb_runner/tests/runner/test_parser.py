import pytest
import argparse
from lcb_runner.runner.parser import get_args
from lcb_runner.utils.scenarios import Scenario

def test_default_args(monkeypatch):
    monkeypatch.setattr("sys.argv", ["script.py"])
    args = get_args()
    assert args.model == "gpt-3.5-turbo-0301"
    assert args.scenario == Scenario.codegeneration
    assert args.n == 10
    assert args.temperature == 0.2
    assert args.top_p == 0.95
    assert args.max_tokens == 2000
    assert args.stop == ["###"]
    assert not args.debug
    assert not args.evaluate

def test_custom_args(monkeypatch):
    monkeypatch.setattr("sys.argv", [
        "script.py",
        "--model", "gpt-4",
        "--scenario", "testoutputprediction",
        "--n", "5",
        "--temperature", "0.8",
        "--debug"
    ])
    args = get_args()
    assert args.model == "gpt-4"
    assert args.scenario == Scenario.testoutputprediction
    assert args.n == 5
    assert args.temperature == 0.8
    assert args.debug

def test_invalid_scenario(monkeypatch):
    monkeypatch.setattr("sys.argv", [
        "script.py",
        "--scenario", "invalid"
    ])
    with pytest.raises(ValueError):
        get_args()

def test_multiple_stop_tokens(monkeypatch):
    monkeypatch.setattr("sys.argv", [
        "script.py",
        "--stop", "###,END,STOP"
    ])
    args = get_args()
    assert args.stop == ["###", "END", "STOP"]

def test_multiprocess_auto(monkeypatch):
    monkeypatch.setattr("sys.argv", [
        "script.py",
        "--multiprocess", "-1"
    ])
    args = get_args()
    import os
    assert args.multiprocess == os.cpu_count()
