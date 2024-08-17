import time

import google.generativeai as genai
from typing import List, Optional
import random

from src.data_preparation.gemini_finetuning_data import GeminiFinetuningData
from src.model_tuning.base_model_tuner import BaseModelHandler


class ModelTuner(BaseModelHandler):
    def __init__(self):
        super().__init__()

    def tune_model(self, tuning_data: List[GeminiFinetuningData], name: Optional[str] = None):
        """Tune the Gemini model with the provided data."""
        self.logger.info("Starting model tuning process...")
        
        if name is None:
            name = f'generate-num-{random.randint(0,10000)}'
        
        # Convert tuning data to Gemini API format
        gemini_format_data = [GeminiFinetuningData.to_gemini_format(data) for data in tuning_data]


        # Start the tuning process
        try:
            operation = genai.create_tuned_model(
                display_name=name,
                source_model="models/gemini-1.5-flash-001-tuning",
                training_data=gemini_format_data,
            )
            self.logger.info(f"Tuning job started.")
            return operation
        except Exception as e:
            self.logger.error(f"Error starting tuning job: {str(e)}")
            raise

    def wait_for_tuning_completion(self, operation):
        """Wait for the tuning process to complete."""
        for status in operation.wait_bar():
            self.logger.info(f"Tuning status: {status}")
            time.sleep(60)

        result = operation.result()
        self.logger.info("Tuning completed successfully.")
        return result