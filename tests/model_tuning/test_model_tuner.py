import pytest
from unittest.mock import Mock, patch
from src.model_tuning.model_tuner import ModelTuner
from src.data_preparation.models import GeminiFinetuningData

class TestModelTuner:
    @pytest.fixture
    def mock_model(self):
        model = Mock()
        model.supported_generation_methods = ["createTunedModel"]
        model.name = "test_model"
        return model

    @pytest.fixture
    def tuning_data(self):
        return [{"text_input": "Hello", "output": "World"}]

    @patch('google.generativeai.list_models')
    @patch('google.generativeai.configure')
    def test_setup_model(self, mock_configure, mock_list_models, mock_model):
        mock_list_models.return_value = [mock_model]
        
        tuner = ModelTuner()
        mock_configure.assert_called_once()
        assert tuner.model == mock_model

    @patch('google.generativeai.create_tuned_model')
    def test_tune_model(self, mock_create_tuned_model, mock_model, tuning_data):
        tuner = ModelTuner()
        tuner.model = mock_model
        mock_tuning_job = Mock()
        mock_create_tuned_model.return_value = mock_tuning_job

        result = tuner.tune_model(tuning_data)
        
        mock_create_tuned_model.assert_called_once_with(
            source_model=tuner.model.name,
            training_data=tuning_data,
            epoch_count=3,
            batch_size=32,
            learning_rate=3e-4,
            warmup_steps=100
        )
        assert result == mock_tuning_job
        assert tuner.tuning_job == mock_tuning_job

    @patch('google.generativeai.get_tuned_model')
    def test_get_tuned_model_status(self, mock_get_tuned_model):
        tuner = ModelTuner()
        tuner.tuning_job = Mock(name="test_job")
        mock_tuned_model = Mock(status="ACTIVE")
        mock_get_tuned_model.return_value = mock_tuned_model

        status = tuner.get_tuned_model_status()
        
        mock_get_tuned_model.assert_called_once_with("test_job")
        assert status == "ACTIVE"

    @patch('src.model_tuning.model_tuner.ModelTuner.get_tuned_model_status')
    @patch('google.generativeai.get_tuned_model')
    @patch('time.sleep')
    def test_wait_for_tuning_completion(self, mock_sleep, mock_get_tuned_model, mock_get_status):
        tuner = ModelTuner()
        tuner.tuning_job = Mock(name="test_job")
        mock_tuned_model = Mock()
        mock_get_tuned_model.return_value = mock_tuned_model

        # Simulate status changing from "RUNNING" to "ACTIVE"
        mock_get_status.side_effect = ["RUNNING", "ACTIVE"]

        result = tuner.wait_for_tuning_completion()

        assert mock_get_status.call_count == 2
        mock_sleep.assert_called_once_with(60)
        assert result == mock_tuned_model

    @patch('src.model_tuning.model_tuner.ModelTuner.get_tuned_model_status')
    def test_wait_for_tuning_completion_failure(self, mock_get_status):
        tuner = ModelTuner()
        tuner.tuning_job = Mock(name="test_job")

        mock_get_status.return_value = "FAILED"

        with pytest.raises(Exception, match="Tuning failed with status: FAILED"):
            tuner.wait_for_tuning_completion()

    def test_get_tuned_model_status_no_job(self):
        tuner = ModelTuner()
        with pytest.raises(ValueError, match="No tuning job has been started"):
            tuner.get_tuned_model_status()

    def test_wait_for_tuning_completion_no_job(self):
        tuner = ModelTuner()
        with pytest.raises(ValueError, match="No tuning job has been started"):
            tuner.wait_for_tuning_completion()