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

3. Set up your Google API key:
   Create a `.env` file in the project root and add your API key:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

## Usage

To run the tuning process:

```
python scripts/run_tuning.py --data_file path/to/training_data.csv --test_file path/to/test_data.csv --output_model path/to/save/tuned_model --output_results path/to/save/evaluation_results.json
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