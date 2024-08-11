import click
from src.data_preparation import load_data, preprocess_data, format_data_for_tuning
from src.model_tuning import setup_model, tune_model, save_tuned_model
from src.evaluation import evaluate_model
from src.utils import save_results
from config.logging_config import setup_logging
import logging

logger = logging.getLogger(__name__)

@click.command()
@click.option('--data_file', required=True, help="Path to the training data CSV file")
@click.option('--test_file', required=True, help="Path to the test data CSV file")
@click.option('--output_model', required=True, help="Path to save the tuned model")
@click.option('--output_results', required=True, help="Path to save the evaluation results")
def main(data_file, test_file, output_model, output_results):
    setup_logging()
    logger.info("Starting Gemini model tuning process")

    # Load and preprocess data
    df = load_data(data_file)
    preprocessed_df = preprocess_data(df)
    tuning_data = format_data_for_tuning(preprocessed_df)

    # Set up and tune the model
    model = setup_model()
    tuned_model = tune_model(model, tuning_data)

    # Save the tuned model
    save_tuned_model(tuned_model, output_model)

    # Evaluate the tuned model
    test_data = load_data(test_file)
    evaluation_results = evaluate_model(tuned_model, test_data)

    # Save evaluation results
    save_results(evaluation_results, output_results)

    logger.info("Gemini model tuning process completed")

if __name__ == "__main__":
    main()