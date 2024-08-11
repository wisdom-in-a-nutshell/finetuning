import google.generativeai as genai
from typing import List
import time
import logging

from config.settings import API_KEY, MODEL_NAME
from src.data_preparation.models import GeminiFinetuningData

class ModelTuner:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.tuning_job = None
        self.setup_model()

    def setup_model(self):
        """Set up the Gemini model for tuning."""
        genai.configure(api_key=API_KEY)
        all_models = genai.list_models()
        self.logger.info(f"Available models in Gemini: {[m.name for m in all_models]}")
        models = [m for m in all_models if "createTunedModel" in m.supported_generation_methods]
        if not models:
            raise ValueError("No suitable models found for tuning")
        self.model = models[0]

    def tune_model(self, tuning_data: List[GeminiFinetuningData]):
        """Tune the Gemini model with the provided data."""
        self.logger.info("Starting model tuning process...")
        
        # Start the tuning process
        self.tuning_job = genai.create_tuned_model(
            source_model=self.model.name,
            training_data=tuning_data,
            # Using default hyperparameters as per Gemini documentation
            epoch_count=3,
            batch_size=32,
            learning_rate=3e-4,
            warmup_steps=100
        )
        
        self.logger.info(f"Tuning job started. Model name: {self.tuning_job.name}")
        return self.tuning_job

    def get_tuned_model_status(self):
        """Check the status of the tuning job."""
        if not self.tuning_job:
            raise ValueError("No tuning job has been started")
        model = genai.get_tuned_model(self.tuning_job.name)
        return model.status

    def wait_for_tuning_completion(self):
        """Wait for the tuning process to complete."""
        if not self.tuning_job:
            raise ValueError("No tuning job has been started")
        while True:
            status = self.get_tuned_model_status()
            self.logger.info(f"Tuning status: {status}")
            if status == "ACTIVE":
                self.logger.info("Tuning completed successfully.")
                return genai.get_tuned_model(self.tuning_job.name)
            elif status in ["FAILED", "CANCELLED"]:
                raise Exception(f"Tuning failed with status: {status}")
            time.sleep(60)  # Check status every minute