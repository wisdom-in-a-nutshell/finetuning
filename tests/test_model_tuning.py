import pytest
from unittest.mock import Mock, patch
from src.model_tuning import setup_model, tune_model, save_tuned_model

@patch('google.generativeai.configure')
@patch('google.generativeai.GenerativeModel')
def test_setup_model(mock_generative_model, mock_configure):
    model = setup_model()
    mock_configure.assert_called_once()
    mock_generative_model.assert_called_once_with("gemini-1.5-pro")

def test_tune_model():
    mock_model = Mock()
    tuning_data = [{"input_text": "Hello", "output_text": "World"}]
    result = tune_model(mock_model, tuning_data)
    assert result == mock_model  # Assuming the function returns the model itself

def test_save_tuned_model(capsys):
    mock_model = Mock()
    output_path = "path/to/model.pkl"
    save_tuned_model(mock_model, output_path)
    captured = capsys.readouterr()
    assert f"Tuned model would be saved to {output_path}" in captured.out