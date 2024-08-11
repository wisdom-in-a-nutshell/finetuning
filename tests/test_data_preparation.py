import pytest
from src.data_preparation import load_data, format_data_for_tuning

def test_load_data(tmp_path):
    # Create a temporary CSV file
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "test.csv"
    p.write_text("input,output\nHello,World\nTest,Data")
    
    data = load_data(str(p))
    assert len(data) == 2
    assert data[0] == {"input": "Hello", "output": "World"}
    assert data[1] == {"input": "Test", "output": "Data"}

def test_format_data_for_tuning():
    test_data = [
        {"input": "Hello", "output": "World"},
        {"input": "Test", "output": "Data"}
    ]
    formatted_data = format_data_for_tuning(test_data)
    assert len(formatted_data) == 2
    assert formatted_data[0] == {"input_text": "Hello", "output_text": "World"}
    assert formatted_data[1] == {"input_text": "Test", "output_text": "Data"}