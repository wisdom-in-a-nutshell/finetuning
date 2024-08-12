import logging
import os
import pytest
import random
from dotenv import load_dotenv
import google.generativeai as genai

from src.data_preparation.gemini_finetuning_data import GeminiFinetuningData
from src.model_tuning.base_model_tuner import BaseModelHandler  # Update this import if necessary

load_dotenv()

class TestModelTunerIntegration:
    @pytest.fixture(scope="class")
    def model_tuner(self):
        client_secret_path = os.getenv('CLIENT_SECRET_PATH')
        if not client_secret_path:
            pytest.skip("CLIENT_SECRET_PATH not set in environment or .env file")
        tuner = BaseModelHandler()
        assert tuner.creds is not None, "Credentials not set up properly in ModelTuner"
        return tuner

    def test_setup_credentials_integration(self, model_tuner):
        assert model_tuner.creds is not None

    def test_tune_model_integration(self, model_tuner):
        tuning_data = [
            GeminiFinetuningData(text_input="1", output="2"),
            GeminiFinetuningData(text_input="3", output="4"),
            GeminiFinetuningData(text_input="-3", output="-2"),
            GeminiFinetuningData(text_input="twenty two", output="twenty three"),
            GeminiFinetuningData(text_input="two hundred", output="two hundred one"),
            GeminiFinetuningData(text_input="ninety nine", output="one hundred"),
        ]

        name = f'generate-num-{random.randint(0,10000)}'

        tuning_job = model_tuner.tune_model(tuning_data, name)

        assert tuning_job is not None
        assert hasattr(tuning_job, 'name')

        try:
            tuned_model = model_tuner.wait_for_tuning_completion(tuning_job)
            assert tuned_model is not None

            model = genai.GenerativeModel(model_name=f'tunedModels/{name}')
            result = model.generate_content('55')
            assert result.text == '56'

        except Exception as e:
            pytest.fail(f"Tuning failed: {str(e)}")
        finally:
            logging.info(f"Tuning model done: {name}")

    def test_get_available_models_integration(self, model_tuner):
        available_models = model_tuner.get_available_models()
        
        assert isinstance(available_models, list)
        assert len(available_models) > 0
        
        for model in available_models:
            assert hasattr(model, 'name')
            assert hasattr(model, 'supported_generation_methods')

    def test_get_tuned_model_status_integration(self, model_tuner):
        model_name = os.getenv('TEST_MODEL_NAME')
        if not model_name:
            pytest.skip("TEST_MODEL_NAME not set in environment or .env file")
        
        status = model_tuner.get_tuned_model_status(model_name)
        assert status is not None

    def test_get_tuned_model_integration(self, model_tuner):
        model_name = os.getenv('TEST_MODEL_NAME')
        if not model_name:
            pytest.skip("TEST_MODEL_NAME not set in environment or .env file")
        
        tuned_model = model_tuner.get_tuned_model(model_name)
        assert tuned_model is not None
        assert isinstance(tuned_model, genai.GenerativeModel)

    def test_get_tuned_models_integration(self, model_tuner):
        tuned_models = model_tuner.get_tuned_models()
        
        assert hasattr(tuned_models, '__iter__'), "get_tuned_models should return an iterable"
        
        models_list = list(tuned_models)  # Convert iterable to list for easier testing
        
        for model in models_list:
            assert isinstance(model, dict), "Each tuned model should be a dictionary"
            assert 'name' in model, "Each tuned model should have a 'name' field"
            assert model['name'].startswith('tunedModels/'), "Tuned model name should start with 'tunedModels/'"
            assert 'createTime' in model, "Each tuned model should have a 'createTime' field"
            assert 'baseModel' in model, "Each tuned model should have a 'baseModel' field"
        
        logging.info(f"Found {len(models_list)} tuned models")

    # Add more integration tests as needed