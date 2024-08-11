import json
from typing import List, Dict

def load_data(file_path: str) -> List[Dict[str, List[Dict[str, str]]]]:
    """Load training data from a JSONL file."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def format_data_for_tuning(data: List[Dict[str, List[Dict[str, str]]]]) -> List[Dict[str, str]]:
    """Format the data into the required structure for tuning."""
    formatted_data = []
    for item in data:
        input_text = ""
        output_text = ""
        for message in item["messages"]:
            if message["role"] == "system":
                input_text += message["content"] + " "
            elif message["role"] == "user":
                input_text += message["content"]
            elif message["role"] == "assistant":
                output_text = message["content"]
        
        formatted_data.append({
            "input_text": input_text.strip(),
            "output_text": output_text
        })
    return formatted_data