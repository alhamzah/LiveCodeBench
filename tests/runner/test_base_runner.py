import pytest
from unittest.mock import Mock, patch
from lcb_runner.runner.base_runner import BaseRunner
from lcb_runner.lm_styles import LanguageModel

class TestRunner(BaseRunner):
    def _run_single(self, prompt):
        return ["test response"]

class TestBaseRunner:
    @pytest.fixture
    def mock_args(self):
        args = Mock()
        args.use_cache = False
        args.n = 1
        args.multiprocess = 1
        return args

    @pytest.fixture
    def mock_model(self):
        model = Mock(spec=LanguageModel)
        model.model_repr = "test_model"
        return model

    def test_run_batch_single_prompt(self, mock_args, mock_model):
        # Arrange
        runner = TestRunner(mock_args, mock_model)
        test_prompts = ["test prompt"]

        # Act
        result = runner.run_batch(test_prompts)

        # Assert
        assert len(result) == 1
        assert result[0] == ["test response"]
