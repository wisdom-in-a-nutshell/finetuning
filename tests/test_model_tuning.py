import pytest
from unittest.mock import Mock, patch
from src.model_tuning import setup_model, tune_model, get_tuned_model_status, wait_for_tuning_completion
from src.data_preparation import GeminiFinetuningData

@patch('google.generativeai.list_models')
@patch('google.generativeai.configure')
def test_setup_model(mock_configure, mock_list_models):
    mock_model = Mock()
    mock_model.supported_generation_methods = ["createTunedModel"]
    mock_list_models.return_value = [mock_model]
    
    model = setup_model()
    mock_configure.assert_called_once()
    assert model == mock_model

@patch('google.generativeai.create_tuned_model')
def test_tune_model(mock_create_tuned_model):
    mock_model = Mock()
    mock_model.name = "test_model"
    mock_tuning_job = Mock()
    mock_create_tuned_model.return_value = mock_tuning_job

    tuning_data: List[GeminiFinetuningData] = [{"text_input": "Hello", "output": "World"}]
    result = tune_model(mock_model, tuning_data)
    
    mock_create_tuned_model.assert_called_once_with(
        source_model=mock_model.name,
        training_data=tuning_data,
        epoch_count=3,
        batch_size=32,
        learning_rate=3e-4,
        warmup_steps=100
    )
    assert result == mock_tuning_job

@patch('google.generativeai.get_tuned_model')
def test_get_tuned_model_status(mock_get_tuned_model):
    mock_tuning_job = Mock()
    mock_tuning_job.name = "test_job"
    mock_tuned_model = Mock()
    mock_tuned_model.status = "ACTIVE"
    mock_get_tuned_model.return_value = mock_tuned_model

    status = get_tuned_model_status(mock_tuning_job)
    
    mock_get_tuned_model.assert_called_once_with("test_job")
    assert status == "ACTIVE"

@patch('src.model_tuning.get_tuned_model_status')
@patch('google.generativeai.get_tuned_model')
@patch('time.sleep')
def test_wait_for_tuning_completion(mock_sleep, mock_get_tuned_model, mock_get_status):
    mock_tuning_job = Mock()
    mock_tuning_job.name = "test_job"
    mock_tuned_model = Mock()
    mock_get_tuned_model.return_value = mock_tuned_model

    # Simulate status changing from "RUNNING" to "ACTIVE"
    mock_get_status.side_effect = ["RUNNING", "ACTIVE"]

    result = wait_for_tuning_completion(mock_tuning_job)

    assert mock_get_status.call_count == 2
    mock_sleep.assert_called_once_with(60)
    assert result == mock_tuned_model

@patch('src.model_tuning.get_tuned_model_status')
def test_wait_for_tuning_completion_failure(mock_get_status):
    mock_tuning_job = Mock()
    mock_tuning_job.name = "test_job"

    mock_get_status.return_value = "FAILED"

    with pytest.raises(Exception, match="Tuning failed with status: FAILED"):
        wait_for_tuning_completion(mock_tuning_job)