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

        # Wait for tuning completion
        self.logger.info("Waiting for tuning to complete...")
        result = model_tuner.wait_for_tuning_completion(tuning_operation)

        # Return the model name
        return result.model

def main():
    parser = argparse.ArgumentParser(description="Run Gemini model tuning")
    parser.add_argument("--data_file", required=True, help="Path to the training data JSONL file")
    parser.add_argument("--model_name", required=True, help="Name for the tuned model")
    args = parser.parse_args()

    runner = TuningRunner()
    runner.run(args.data_file, args.model_name)

if __name__ == "__main__":
    main()