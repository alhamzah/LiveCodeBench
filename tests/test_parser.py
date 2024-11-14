import pytest
import torch
import os
from lcb_runner.runner.parser import get_args
from lcb_runner.utils.scenarios import Scenario

def test_default_args(monkeypatch):
    monkeypatch.setattr("sys.argv", ["script"])
    args = get_args()
    assert args.model == "gpt-3.5-turbo-0301"
    assert args.local_model_path is None
    assert args.trust_remote_code is False
    assert args.scenario == Scenario.codegeneration
    assert args.not_fast is False
    assert args.release_version == "release_v1"
    assert args.n == 10
    assert args.temperature == 0.2
    assert args.stop == ["###"]

def test_custom_args(monkeypatch):
    test_args = [
        "script",
        "--model", "gpt-4",
        "--local_model_path", "/path/to/model",
        "--trust_remote_code",
        "--scenario", "testoutputprediction",
        "--n", "5",
        "--temperature", "0.8",
        "--stop", "END,STOP"
    ]
    monkeypatch.setattr("sys.argv", test_args)
    args = get_args()
    assert args.model == "gpt-4"
    assert args.local_model_path == "/path/to/model"
    assert args.trust_remote_code is True
    assert args.scenario == Scenario.testoutputprediction
    assert args.n == 5
    assert args.temperature == 0.8
    assert args.stop == ["END", "STOP"]

def test_multiprocess_auto_detection(monkeypatch):
    test_args = ["script", "--multiprocess", "-1"]
    monkeypatch.setattr("sys.argv", test_args)
    args = get_args()
    assert args.multiprocess == os.cpu_count()

def test_tensor_parallel_size(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr("sys.argv", ["script"])
    args = get_args()
    assert args.tensor_parallel_size == 4