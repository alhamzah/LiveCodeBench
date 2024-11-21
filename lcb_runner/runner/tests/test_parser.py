import pytest
from unittest.mock import patch
import torch
from lcb_runner.runner.parser import get_args
from lcb_runner.utils.scenarios import Scenario

def test_default_args():
    with patch('sys.argv', ['script.py']):
        args = get_args()
        assert args.model == "gpt-3.5-turbo-0301"
        assert args.local_model_path is None
        assert args.trust_remote_code is False
        assert args.scenario == Scenario.codegeneration
        assert args.not_fast is False
        assert args.release_version == "release_v1"
        assert args.cot_code_execution is False
        assert args.n == 10
        assert args.codegen_n == 10
        assert args.temperature == 0.2
        assert args.top_p == 0.95
        assert args.max_tokens == 2000
        assert args.multiprocess == 0
        assert args.stop == ["###"]
        assert args.continue_existing is False
        assert args.continue_existing_with_eval is False
        assert args.use_cache is False
        assert args.cache_batch_size == 100
        assert args.debug is False
        assert args.evaluate is False
        assert args.num_process_evaluate == 12
        assert args.timeout == 6
        assert args.openai_timeout == 45
        assert args.tensor_parallel_size == torch.cuda.device_count()
        assert args.enable_prefix_caching is False
        assert args.custom_output_file is None
        assert args.custom_output_save_name is None
        assert args.dtype == "bfloat16"

def test_custom_args():
    test_args = [
        'script.py',
        '--model', 'test-model',
        '--local_model_path', '/path/to/model',
        '--trust_remote_code',
        '--scenario', 'codegeneration',
        '--not_fast',
        '--release_version', 'test_v1',
        '--cot_code_execution',
        '--n', '5',
        '--codegen_n', '15',
        '--temperature', '0.5',
        '--top_p', '0.8',
        '--max_tokens', '1000',
        '--multiprocess', '4',
        '--stop', 'STOP1,STOP2',
        '--continue_existing',
        '--continue_existing_with_eval',
        '--use_cache',
        '--cache_batch_size', '200',
        '--debug',
        '--evaluate',
        '--num_process_evaluate', '8',
        '--timeout', '10',
        '--openai_timeout', '60',
        '--tensor_parallel_size', '2',
        '--enable_prefix_caching',
        '--custom_output_file', 'output.txt',
        '--custom_output_save_name', 'test_results',
        '--dtype', 'float16'
    ]
    
    with patch('sys.argv', test_args):
        args = get_args()
        assert args.model == "test-model"
        assert args.local_model_path == "/path/to/model"
        assert args.trust_remote_code is True
        assert args.scenario == Scenario.codegeneration
        assert args.not_fast is True
        assert args.release_version == "test_v1"
        assert args.cot_code_execution is True
        assert args.n == 5
        assert args.codegen_n == 15
        assert args.temperature == 0.5
        assert args.top_p == 0.8
        assert args.max_tokens == 1000
        assert args.multiprocess == 4
        assert args.stop == ["STOP1", "STOP2"]
        assert args.continue_existing is True
        assert args.continue_existing_with_eval is True
        assert args.use_cache is True
        assert args.cache_batch_size == 200
        assert args.debug is True
        assert args.evaluate is True
        assert args.num_process_evaluate == 8
        assert args.timeout == 10
        assert args.openai_timeout == 60
        assert args.tensor_parallel_size == 2
        assert args.enable_prefix_caching is True
        assert args.custom_output_file == "output.txt"
        assert args.custom_output_save_name == "test_results"
        assert args.dtype == "float16"

def test_multiprocess_default():
    with patch('sys.argv', ['script.py', '--multiprocess', '-1']):
        with patch('os.cpu_count', return_value=8):
            args = get_args()
            assert args.multiprocess == 8
