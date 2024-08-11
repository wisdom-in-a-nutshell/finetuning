import pytest
import json
import tempfile
from src.data_preparation import (
    validate_openai_chat_format,
    load_and_validate_data,
    format_data_for_gemini,
    prepare_data_for_gemini,
    InvalidDataFormatError,
    InvalidJSONError
)

def test_validate_openai_chat_format():
    valid_data = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there! How can I assist you today?"}
        ]
    }
    assert validate_openai_chat_format(valid_data) == True

    invalid_data = {
        "messages": [
            {"role": "invalid_role", "content": "This is invalid."},
        ]
    }
    assert validate_openai_chat_format(invalid_data) == False

def test_load_and_validate_data():
    valid_data = [
        '{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello!"}, {"role": "assistant", "content": "Hi there!"}]}',
        '{"messages": [{"role": "user", "content": "How are you?"}, {"role": "assistant", "content": "I\'m doing well, thank you!"}]}'
    ]

    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        for line in valid_data:
            temp_file.write(line + '\n')
    
    result = load_and_validate_data(temp_file.name)
    assert len(result) == 2
    assert all(isinstance(item, dict) for item in result)

def test_load_and_validate_data_with_invalid_file():
    invalid_data = [
        '{"invalid": "data"}',
        '{"also": "invalid"}'
    ]

    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        for line in invalid_data:
            temp_file.write(line + '\n')
    
    with pytest.raises(InvalidDataFormatError):
        load_and_validate_data(temp_file.name)

def test_load_and_validate_data_with_invalid_json():
    invalid_data = [
        '{"messages": [{"role": "user", "content": "Hello"}]}',
        '{"messages": [{"role": "assistant", "content": "Hi"}]',  # Missing closing brace
    ]

    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        for line in invalid_data:
            temp_file.write(line + '\n')
    
    with pytest.raises(InvalidJSONError):
        load_and_validate_data(temp_file.name)

def test_load_and_validate_data_with_empty_file():
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        temp_file.write('')  # Create an empty file
    
    with pytest.raises(InvalidDataFormatError):
        load_and_validate_data(temp_file.name)

def test_load_and_validate_data_with_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        load_and_validate_data("nonexistent_file.jsonl")

def test_format_data_for_gemini():
    input_data = [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi there! How can I assist you today?"}
            ]
        }
    ]
    result = format_data_for_gemini(input_data)
    assert len(result) == 1
    assert "input_text" in result[0]
    assert "output_text" in result[0]
    assert result[0]["input_text"] == "You are a helpful assistant. Hello!"
    assert result[0]["output_text"] == "Hi there! How can I assist you today?"

def test_prepare_data_for_gemini():
    valid_data = [
        '{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello!"}, {"role": "assistant", "content": "Hi there!"}]}',
        '{"messages": [{"role": "user", "content": "How are you?"}, {"role": "assistant", "content": "I\'m doing well, thank you!"}]}'
    ]

    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        for line in valid_data:
            temp_file.write(line + '\n')
    
    result = prepare_data_for_gemini(temp_file.name)
    assert len(result) == 2
    assert all(isinstance(item, dict) for item in result)
    assert all("input_text" in item and "output_text" in item for item in result)

def test_prepare_data_for_gemini_with_invalid_file():
    invalid_data = [
        '{"invalid": "data"}',
        '{"also": "invalid"}'
    ]

    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        for line in invalid_data:
            temp_file.write(line + '\n')
    
    with pytest.raises(InvalidDataFormatError):
        prepare_data_for_gemini(temp_file.name)