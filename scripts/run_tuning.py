import os
import sys
import argparse

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.data_preparation import prepare_data_for_gemini
from src.model_tuning import setup_model, tune_model, save_tuned_model
from src.evaluation import evaluate_model, save_results

def main(data_file, test_file, output_model, output_results):
    print("Starting Gemini model tuning process")

    # Load and preprocess data
    tuning_data = prepare_data_for_gemini(data_file)

    # Set up and tune the model
    model = setup_model()
    tuned_model = tune_model(model, tuning_data)

    # Save the tuned model
    save_tuned_model(tuned_model, output_model)

    # Evaluate the tuned model
    test_data = prepare_data_for_gemini(test_file)
    evaluation_results = evaluate_model(tuned_model, test_data)

    # Save evaluation results
    save_results(evaluation_results, output_results)

    print("Gemini model tuning process completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Gemini model tuning and evaluation")
    parser.add_argument("--data_file", required=True, help="Path to the training data JSONL file")
    parser.add_argument("--test_file", required=True, help="Path to the test data JSONL file")
    parser.add_argument("--output_model", required=True, help="Path to save the tuned model")
    parser.add_argument("--output_results", required=True, help="Path to save the evaluation results")
    args = parser.parse_args()
    main(args.data_file, args.test_file, args.output_model, args.output_results)