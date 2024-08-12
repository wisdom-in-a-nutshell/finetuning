# Gemini Model Tuning

This project demonstrates how to fine-tune (tune) a Gemini model using the Google Generative AI API.

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/your-username/gemini-tuning.git
   cd gemini-tuning
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. Set up your Google API credentials:
   - Go to the [Google Cloud Console](https://console.cloud.google.com/).
   - Create a new project or select an existing one.
   - Enable the Generative AI API for your project.
   - Create a service account and download the JSON key file.
   - Rename the key file to `service_account.json` and place it in the project root directory.
   - Offical detailed instructions are [here]([https://console.cloud.google.com/](https://ai.google.dev/gemini-api/docs/oauth))


4. Set the environment variable for authentication:
   ```
   export GOOGLE_APPLICATION_CREDENTIALS="path/to/service_account.json"
   ```
   On Windows, use:
   ```
   set GOOGLE_APPLICATION_CREDENTIALS=path\to\service_account.json
   ```

## Usage

To run the tuning process:

```
python scripts/run_tuning.py --data_file path/to/training_data.jsonl --test_file path/to/test_data.jsonl --output_model path/to/save/tuned_model --output_results path/to/save/evaluation_results.json
```

## Project Structure

- `src/`: Contains the main source code for data preparation, model tuning, and evaluation.
- `scripts/`: Contains the main script to run the tuning process.
- `tests/`: Contains unit tests for the project components.
- `config.py`: Configuration file for API keys and model settings.
- `requirements.txt`: Lists the required dependencies for the project.
- `.gitignore`: Specifies files and directories to ignore in the Git repository.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
