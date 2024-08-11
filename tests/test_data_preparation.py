import pytest
import tempfile
from src.data_preparation.data_preparator import DataPreparator
from src.data_preparation.exceptions import InvalidDataFormatError, InvalidJSONError

def test_validate_openai_chat_format():
    preparator = DataPreparator("dummy_path.jsonl")
    valid_data = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there! How can I assist you today?"}
        ]
    }
    assert preparator.validate_openai_chat_format(valid_data) == True

    invalid_data = {
        "messages": [
            {"role": "invalid_role", "content": "This is invalid."},
        ]
    }
    assert preparator.validate_openai_chat_format(invalid_data) == False

def test_load_and_validate_data():
    valid_data = [
        '{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello!"}, {"role": "assistant", "content": "Hi there!"}]}',
        '{"messages": [{"role": "user", "content": "How are you?"}, {"role": "assistant", "content": "I\'m doing well, thank you!"}]}'
    ]

    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        for line in valid_data:
            temp_file.write(line + '\n')
    
    preparator = DataPreparator(temp_file.name)
    result = preparator.load_and_validate_data()
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
    
    preparator = DataPreparator(temp_file.name)
    with pytest.raises(InvalidDataFormatError):
        preparator.load_and_validate_data()

def test_load_and_validate_data_with_invalid_json():
    invalid_data = [
        '{"messages": [{"role": "user", "content": "Hello"}]}',
        '{"messages": [{"role": "assistant", "content": "Hi"}]',  # Missing closing brace
    ]

    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        for line in invalid_data:
            temp_file.write(line + '\n')
    
    preparator = DataPreparator(temp_file.name)
    with pytest.raises(InvalidJSONError):
        preparator.load_and_validate_data()

def test_load_and_validate_data_with_empty_file():
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        temp_file.write('')  # Create an empty file
    
    preparator = DataPreparator(temp_file.name)
    with pytest.raises(InvalidDataFormatError):
        preparator.load_and_validate_data()

def test_load_and_validate_data_with_nonexistent_file():
    preparator = DataPreparator("nonexistent_file.jsonl")
    with pytest.raises(FileNotFoundError):
        preparator.load_and_validate_data()

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
    preparator = DataPreparator("dummy_path.jsonl")
    result = preparator.format_data_for_gemini(input_data)
    assert len(result) == 1
    assert "text_input" in result[0]
    assert "output" in result[0]
    assert result[0]["text_input"] == "You are a helpful assistant. Hello!"
    assert result[0]["output"] == "Hi there! How can I assist you today?"

def test_prepare_data():
    valid_data = [
        '{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello!"}, {"role": "assistant", "content": "Hi there!"}]}',
        '{"messages": [{"role": "user", "content": "How are you?"}, {"role": "assistant", "content": "I\'m doing well, thank you!"}]}'
    ]

    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        for line in valid_data:
            temp_file.write(line + '\n')
    
    preparator = DataPreparator(temp_file.name)
    result = preparator.prepare_data()
    assert len(result) == 2
    assert all(isinstance(item, dict) for item in result)
    assert all("text_input" in item and "output" in item for item in result)

def test_prepare_data_with_invalid_file():
    invalid_data = [
        '{"invalid": "data"}',
        '{"also": "invalid"}'
    ]

    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        for line in invalid_data:
            temp_file.write(line + '\n')
    
    preparator = DataPreparator(temp_file.name)
    with pytest.raises(InvalidDataFormatError):
        preparator.prepare_data()