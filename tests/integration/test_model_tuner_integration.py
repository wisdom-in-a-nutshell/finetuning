import os
import pytest
import random
from dotenv import load_dotenv
import google.generativeai as genai
from src.model_tuning.model_tuner import ModelTuner

@pytest.fixture(scope="module")
def model_tuner():
    load_dotenv()  # This will load the variables from .env file
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        pytest.skip("GEMINI_API_KEY not set in environment or .env file")
    genai.configure(api_key=api_key)
    return ModelTuner()

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
        tuned_model = model_tuner.wait_for_tuning_completion()
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