import csv
from typing import List, Dict

def load_data(file_path: str) -> List[Dict[str, str]]:
    """Load training data from a CSV file."""
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)

def format_data_for_tuning(data: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Format the data into the required structure for tuning."""
    return [
        {
            "input_text": row["input"],
            "output_text": row["output"]
        }
        for row in data
    ]