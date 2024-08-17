import os
import sys
import argparse
import logging

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.data_preparation.data_preparator import DataPreparator
from src.model_tuning.model_tuner import ModelTuner

class TuningRunner:
    def __init__(self):
        self.logger = self.setup_logging()

    @staticmethod
    def setup_logging():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        return logging.getLogger(__name__)

    def run(self, data_file, model_name):
        self.logger.info("Starting Gemini model tuning process")

        # Prepare data
        self.logger.info(f"Preparing data from {data_file}")
        data_preparator = DataPreparator(data_file)
        tuning_data = data_preparator.prepare_data()

        # Set up and tune the model
        self.logger.info("Setting up ModelTuner")
        model_tuner = ModelTuner()

        self.logger.info("Starting model tuning")
        tuning_operation = model_tuner.tune_model(tuning_data, name=model_name)

        # Return the model name
        return tuning_operation.metadata.name