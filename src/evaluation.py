import json

def save_results(results, output_file):
    """Save evaluation results to a JSON file."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)