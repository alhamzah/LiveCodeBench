import pytest
import torch
import os
from unittest.mock import patch

from lcb_runner.runner.parser import get_args
from lcb_runner.utils.scenarios import Scenario

def test_default_args():
    with patch('sys.argv', ['script.py']):
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

def test_custom_args():
    test_args = [
        'script.py',
        '--model', 'gpt-4',
        '--scenario', 'testoutputprediction',
        '--n', '5',
        '--temperature', '0.8',
        '--top_p', '0.9',
        '--max_tokens', '1000',
        '--stop', 'END,STOP',
        '--debug',
        '--evaluate'
    ]
    with patch('sys.argv', test_args):
        args = get_args()
        assert args.model == "gpt-4"
        assert args.scenario == Scenario.testoutputprediction
        assert args.n == 5
        assert args.temperature == 0.8
        assert args.top_p == 0.9
        assert args.max_tokens == 1000
        assert args.stop == ["END", "STOP"]
        assert args.debug
        assert args.evaluate

def test_invalid_scenario():
    test_args = ['script.py', '--scenario', 'invalid']
    with patch('sys.argv', test_args):
        with pytest.raises(ValueError):
            get_args()

def test_multiprocess_auto_detection():
    test_args = ['script.py', '--multiprocess', '-1']
    with patch('sys.argv', test_args):
        args = get_args()
        assert args.multiprocess == os.cpu_count()

def test_tensor_parallel_size_auto_detection():
    test_args = ['script.py', '--tensor_parallel_size', '-1']
    with patch('sys.argv', test_args):
        args = get_args()
        assert args.tensor_parallel_size == torch.cuda.device_count()

[tool.poetry.dev-dependencies]
pytest = "^7.0.0"

__pycache__/
.pytest_cache/
