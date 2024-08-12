import pytest
from unittest.mock import patch, MagicMock
from src.model_tuning.model_tuner import ModelTuner
import google.generativeai as genai
from google.ai import generativelanguage as glm
from dotenv import load_dotenv

class TestModelTuner:
    @pytest.fixture(autouse=True)
    def setup(self):
        # Load environment variables from .env file
        load_dotenv()
        self.tuner = ModelTuner()

    def test_get_tuned_model_status_unit(self):
        with patch('google.generativeai.get_model') as mock_get_model:
            # Test ACTIVE state
            mock_model = MagicMock()
            mock_model.state = glm.TunedModel.State.ACTIVE
            mock_get_model.return_value = mock_model
            
            status = self.tuner.get_tuned_model_status("test_model_active")
            assert status == "Ready for use"
            
            # Test CREATING state
            mock_model.state = glm.TunedModel.State.CREATING
            status = self.tuner.get_tuned_model_status("test_model_creating")
            assert status == "Still being created"
            
            # Test FAILED state
            mock_model.state = glm.TunedModel.State.FAILED
            status = self.tuner.get_tuned_model_status("test_model_failed")
            assert status == "Creation failed"
            
            # Test unknown state
            mock_model.state = MagicMock()
            mock_model.state.name = "UNKNOWN"
            status = self.tuner.get_tuned_model_status("test_model_unknown")
            assert status == "Unknown state: UNKNOWN"
            

    def test_get_tuned_model_status_error_handling(self):
        with patch('google.generativeai.get_model', side_effect=Exception("API Error")):
            with pytest.raises(Exception):
                self.tuner.get_tuned_model_status("error_model")

    # Add more unit tests as needed