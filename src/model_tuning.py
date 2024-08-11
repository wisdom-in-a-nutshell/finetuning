import google.generativeai as genai
from config.settings import API_KEY, MODEL_NAME, TUNED_MODEL_NAME
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

def setup_model():
    """Set up the Gemini model for tuning."""
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel(MODEL_NAME)
    return model

def tune_model(model: genai.GenerativeModel, tuning_data: List[Dict[str, str]]):
    """Tune the Gemini model with the provided data."""
    try:
        tuning_config = genai.tuning.TuningConfig(
            model=model,
            tuning_data=tuning_data,
            tuned_model_name=TUNED_MODEL_NAME
        )
        tuned_model = genai.tuning.tune_model(tuning_config)
        return tuned_model
    except Exception as e:
        logger.error(f"Error during model tuning: {str(e)}")
        raise

def save_tuned_model(tuned_model, output_path: str):
    """Save the tuned model to a file."""
    try:
        tuned_model.save(output_path)
        logger.info(f"Tuned model saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving tuned model: {str(e)}")
        raise