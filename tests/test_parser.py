import pytest
from lcb_runner.runner.parser import get_args
from lcb_runner.utils.scenarios import Scenario

def test_default_args():
    # Test with no command line arguments
    args = get_args()
    
    # Verify default values
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
    assert args.enable_prefix_caching is False
    assert args.custom_output_file is None
    assert args.custom_output_save_name is None
    assert args.dtype == "bfloat16"
