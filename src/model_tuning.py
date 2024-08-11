import google.generativeai as genai
from typing import List, Dict
import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from config.settings import API_KEY, MODEL_NAME, TUNED_MODEL_NAME

def setup_model():
    """Set up the Gemini model for tuning."""
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel(MODEL_NAME)
    return model

def tune_model(model: genai.GenerativeModel, tuning_data: List[Dict[str, str]]):
    """Tune the Gemini model with the provided data."""
    tuning_config = genai.tuning.TuningConfig(
        model=model,
        tuning_data=tuning_data,
        tuned_model_name=TUNED_MODEL_NAME
    )
    tuned_model = genai.tuning.tune_model(tuning_config)
    return tuned_model

def save_tuned_model(tuned_model, output_path: str):
    """Save the tuned model to a file."""
    tuned_model.save(output_path)
    print(f"Tuned model saved to {output_path}")