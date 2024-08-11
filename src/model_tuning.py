import google.generativeai as genai
from typing import List
import os
import sys
import time

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from config.settings import API_KEY, MODEL_NAME
from src.data_preparation import GeminiFinetuningData

def setup_model():
    """Set up the Gemini model for tuning."""
    genai.configure(api_key=API_KEY)
    models = [m for m in genai.list_models() if "createTunedModel" in m.supported_generation_methods]
    if not models:
        raise ValueError("No suitable models found for tuning")
    return models[0]

def tune_model(model: genai.Model, tuning_data: List[GeminiFinetuningData]):
    """Tune the Gemini model with the provided data."""
    print("Starting model tuning process...")
    
    # Start the tuning process
    tuning_job = genai.create_tuned_model(
        source_model=model.name,
        training_data=tuning_data,
        # Using default hyperparameters as per Gemini documentation
        epoch_count=3,
        batch_size=32,
        learning_rate=3e-4,
        warmup_steps=100
    )
    
    print(f"Tuning job started. Model name: {tuning_job.name}")
    return tuning_job

def get_tuned_model_status(tuning_job):
    """Check the status of the tuning job."""
    model = genai.get_tuned_model(tuning_job.name)
    return model.status

def wait_for_tuning_completion(tuning_job):
    """Wait for the tuning process to complete."""
    while True:
        status = get_tuned_model_status(tuning_job)
        print(f"Tuning status: {status}")
        if status == "ACTIVE":
            print("Tuning completed successfully.")
            return genai.get_tuned_model(tuning_job.name)
        elif status in ["FAILED", "CANCELLED"]:
            raise Exception(f"Tuning failed with status: {status}")
        time.sleep(60)  # Check status every minute