import pytest
import json
from src.evaluation import save_results

def test_save_results(tmp_path):
    results = {"accuracy": 0.95, "f1_score": 0.92}
    output_file = tmp_path / "results.json"
    save_results(results, str(output_file))
    
    with open(output_file, 'r') as f:
        saved_results = json.load(f)
    
    assert saved_results == results