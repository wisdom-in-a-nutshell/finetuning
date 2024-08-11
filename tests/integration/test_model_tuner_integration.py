import os
import pytest
import random
from dotenv import load_dotenv
import google.generativeai as genai
from src.model_tuning.model_tuner import ModelTuner

@pytest.fixture(scope="module")
def model_tuner():
    load_dotenv()  # This will load the variables from .env file
    client_secret_path = os.getenv('CLIENT_SECRET_PATH')
    if not client_secret_path:
        pytest.skip("CLIENT_SECRET_PATH not set in environment or .env file")
    tuner = ModelTuner()
    assert tuner.creds is not None, "Credentials not set up properly in ModelTuner"
    return tuner

@pytest.mark.integration
def test_setup_model_integration(model_tuner):
    assert model_tuner.model is not None

@pytest.mark.integration
def test_tune_model_integration(model_tuner):
    # Prepare some sample data for tuning
    tuning_data = [
        {"text_input": "1", "output": "2"},
        {"text_input": "3", "output": "4"},
        {"text_input": "-3", "output": "-2"},
        {"text_input": "twenty two", "output": "twenty three"},
        {"text_input": "two hundred", "output": "two hundred one"},
        {"text_input": "ninety nine", "output": "one hundred"},
    ]

    # Generate a unique name for the tuned model
    name = f'generate-num-{random.randint(0,10000)}'

    # Start the tuning process
    tuning_job = model_tuner.tune_model(tuning_data, name)

    # Check if the tuning job was created successfully
    assert tuning_job is not None
    assert hasattr(tuning_job, 'name')

    # Wait for the tuning to complete (this may take a while)
    try:
        tuned_model = model_tuner.wait_for_tuning_completion(tuning_job)
        assert tuned_model is not None
        assert tuned_model.state == "ACTIVE"

        # Test the tuned model
        model = genai.GenerativeModel(model_name=f'tunedModels/{name}')
        result = model.generate_content('55')
        assert result.text == '56'

    except Exception as e:
        pytest.fail(f"Tuning failed: {str(e)}")
    finally:
        # Clean up: delete the tuned model
        genai.delete_tuned_model(f'tunedModels/{name}')

@pytest.mark.integration
def test_get_available_models_integration(model_tuner):
    # Ensure credentials are set up
    assert model_tuner.creds is not None, "Credentials not set up properly"

    # Get the list of available models
    available_models = model_tuner.get_available_models()
    
    # Check if we got a non-empty list of models
    assert isinstance(available_models, list)
    assert len(available_models) > 0
    
    # Print the names of available models
    print("Available models:")
    for model in available_models:
        print(f"- {model.name}")
        print(f"  Supported generation methods: {model.supported_generation_methods}")

    # Log the names of available models (optional, for informational purposes)
    model_tuner.logger.info("Available models for fine-tuning:")
    for model in available_models:
        model_tuner.logger.info(f"- {model.name}")