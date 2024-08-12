import os
import pytest
from scripts.tuning_runner import TuningRunner
from src.model_tuning.model_tuner import ModelTuner
from dotenv import load_dotenv

@pytest.fixture(scope="module")
def setup_environment():
    load_dotenv()
    client_secret_path = os.getenv('CLIENT_SECRET_PATH')
    if not client_secret_path:
        pytest.skip("CLIENT_SECRET_PATH not set in environment or .env file")

@pytest.mark.integration
def test_tuning_runner_integration(setup_environment, tmp_path):
    # Create a temporary JSONL file with sample data
    data_file = tmp_path / "sample_data.jsonl"
    with open(data_file, "w") as f:
        f.write('{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]}\n')
        f.write('{"messages": [{"role": "user", "content": "How are you?"}, {"role": "assistant", "content": "I\'m doing well, thank you for asking!"}]}\n')

    # Initialize TuningRunner
    runner = TuningRunner()

    # Run the tuning process
    model_name = "test_model_integration"
    try:
        tuned_model_name = runner.run(str(data_file), model_name)

        # Verify that the tuned model was created
        assert tuned_model_name is not None
        assert model_name in tuned_model_name

        # Try to get the tuned model status (this will verify if it exists)
        model_tuner = ModelTuner()
        status = model_tuner.get_tuned_model_status(tuned_model_name)
        assert status is not None

    except Exception as e:
        pytest.fail(f"Tuning process failed: {str(e)}")

    finally:
        # Clean up: You might want to delete the tuned model here
        # Be cautious with this in a real environment
        # genai.delete_tuned_model(tuned_model_name)
        pass

@pytest.mark.integration
def test_tuning_runner_with_predetermined_file(setup_environment):
    # Use a predetermined file path
    data_file = "/Users/adi/Downloads/editor_aug12_10.jsonl"

    # Initialize TuningRunner
    runner = TuningRunner()

    # Run the tuning process
    model_name = "video_editing_model_v0.2"
    try:
        tuned_model_name = runner.run(data_file, model_name)

        # Verify that the tuned model was created
        assert tuned_model_name is not None
        assert model_name in tuned_model_name

        # Try to get the tuned model status (this will verify if it exists)
        model_tuner = ModelTuner()
        status = model_tuner.get_tuned_model_status(tuned_model_name)
        assert status is not None

    except Exception as e:
        pytest.fail(f"Tuning process failed with predetermined file: {str(e)}")

    finally:
        # Clean up: You might want to delete the tuned model here
        # Be cautious with this in a real environment
        # genai.delete_tuned_model(tuned_model_name)
        pass