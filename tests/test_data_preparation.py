import pytest
import json
from src.data_preparation import load_data, format_data_for_tuning

def test_load_data(tmp_path):
    # Create a temporary JSONL file
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "test.jsonl"
    data = [
        {"messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there! How can I assist you today?"}
        ]},
        {"messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the weather like?"},
            {"role": "assistant", "content": "I'm sorry, but I don't have access to real-time weather information. Could you please check a weather website or app for the most up-to-date forecast?"}
        ]}
    ]
    with open(p, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    
    loaded_data = load_data(str(p))
    assert len(loaded_data) == 2
    assert loaded_data[0] == data[0]
    assert loaded_data[1] == data[1]

def test_format_data_for_tuning():
    test_data = [
        {"messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there! How can I assist you today?"}
        ]},
        {"messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the weather like?"},
            {"role": "assistant", "content": "I'm sorry, but I don't have access to real-time weather information. Could you please check a weather website or app for the most up-to-date forecast?"}
        ]}
    ]
    formatted_data = format_data_for_tuning(test_data)
    assert len(formatted_data) == 2
    assert formatted_data[0] == {
        "input_text": "You are a helpful assistant. Hello",
        "output_text": "Hi there! How can I assist you today?"
    }
    assert formatted_data[1] == {
        "input_text": "You are a helpful assistant. What's the weather like?",
        "output_text": "I'm sorry, but I don't have access to real-time weather information. Could you please check a weather website or app for the most up-to-date forecast?"
    }