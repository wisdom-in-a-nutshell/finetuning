import pytest
from unittest.mock import patch, MagicMock
from src.model_tuning.model_tuner import ModelTuner
import google.generativeai as genai
from dotenv import load_dotenv
import os
from google.ai import generativelanguage as glm

# Load environment variables from .env file
load_dotenv()

# Unit test with mocking
def test_get_tuned_model_status_unit():
    tuner = ModelTuner()
    
    # Mock the genai.get_model function
    with patch('google.generativeai.get_model') as mock_get_model:
        # Test ACTIVE state
        mock_model = MagicMock()
        mock_model.state = glm.Model.State.ACTIVE
        mock_get_model.return_value = mock_model
        
        status = tuner.get_tuned_model_status("test_model_active")
        assert status == "Ready for use"
        
        # Test CREATING state
        mock_model.state = glm.Model.State.CREATING
        status = tuner.get_tuned_model_status("test_model_creating")
        assert status == "Still being created"
        
        # Test FAILED state
        mock_model.state = glm.Model.State.FAILED
        status = tuner.get_tuned_model_status("test_model_failed")
        assert status == "Creation failed"
        
        # Test DELETING state
        mock_model.state = glm.Model.State.DELETING
        status = tuner.get_tuned_model_status("test_model_deleting")
        assert status == "Being deleted"
        
        # Test UPDATING state
        mock_model.state = glm.Model.State.UPDATING
        status = tuner.get_tuned_model_status("test_model_updating")
        assert status == "Being updated"
        
        # Test unknown state
        mock_model.state = MagicMock()
        mock_model.state.name = "UNKNOWN"
        status = tuner.get_tuned_model_status("test_model_unknown")
        assert status == "Unknown state: UNKNOWN"
        
        # Test NotFoundError
        mock_get_model.side_effect = genai.NotFoundError("Model not found")
        status = tuner.get_tuned_model_status("non_existent_model")
        assert status == "Not found"

# Integration test (requires actual API access)
@pytest.mark.integration
def test_get_tuned_model_status_integration():
    # Ensure the CLIENT_SECRET_PATH is set
    client_secret_path = os.getenv('CLIENT_SECRET_PATH')
    if not client_secret_path:
        pytest.skip("CLIENT_SECRET_PATH not set in .env file")

    tuner = ModelTuner()
    
    # Use an environment variable for the actual model name
    actual_model_name = "tunedModels/videoeditingmodelv01-l32phspsr55s"
    
    status = tuner.get_tuned_model_status(actual_model_name)
    
    # Update assertion to include potential failure reasons
    assert (status in ["Ready for use", "Still being created", "Being deleted", "Being updated"] or
            status.startswith("Creation failed:") or
            status.startswith("Unknown state:"))

# Test error handling
def test_get_tuned_model_status_error_handling():
    tuner = ModelTuner()
    
    with patch('google.generativeai.get_model', side_effect=Exception("API Error")):
        with pytest.raises(Exception):
            tuner.get_tuned_model_status("error_model")

# New test to check if environment variables are loaded correctly
def test_environment_variables():
    assert os.getenv('CLIENT_SECRET_PATH') is not None, "CLIENT_SECRET_PATH should be set in .env file"
    assert os.getenv('TEST_MODEL_NAME') is not None, "TEST_MODEL_NAME should be set in .env file"