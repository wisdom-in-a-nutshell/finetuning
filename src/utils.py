import json

def save_results(results, output_file):
    """Save evaluation results to a JSON file."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

def load_json(file_path):
    """Load JSON data from a file."""
    with open(file_path, 'r') as f:
        return json.load(f)