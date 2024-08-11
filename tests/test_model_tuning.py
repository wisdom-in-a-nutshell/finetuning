import pytest
from unittest.mock import Mock, patch
from src.model_tuning import setup_model, tune_model, save_tuned_model

@patch('google.generativeai.configure')
@patch('google.generativeai.GenerativeModel')
def test_setup_model(mock_generative_model, mock_configure):
    model = setup_model()
    mock_configure.assert_called_once()
    mock_generative_model.assert_called_once_with("gemini-1.5-pro")

@patch('google.generativeai.tuning.tune_model')
def test_tune_model(mock_tune_model):
    mock_model = Mock()
    mock_tuned_model = Mock()
    mock_tune_model.return_value = mock_tuned_model

    tuning_data = [{"input_text": "Hello", "output_text": "World"}]
    result = tune_model(mock_model, tuning_data)

    mock_tune_model.assert_called_once()
    assert result == mock_tuned_model

def test_save_tuned_model(tmp_path):
    mock_model = Mock()
    output_path = tmp_path / "model.pkl"
    save_tuned_model(mock_model, str(output_path))
    mock_model.save.assert_called_once_with(str(output_path))