import pytest
import tempfile
import json
import os
from src.data_preparation.data_preparator import DataPreparator
from src.data_preparation.exceptions import InvalidDataFormatError, InvalidJSONError
from src.data_preparation.chat_message_formatters import OpenAIChatFormat, GeminiFinetuningData

class TestDataPreparator:
    @pytest.fixture
    def valid_jsonl_file(self):
        data = [
            {"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello!"}, {"role": "assistant", "content": "Hi there!"}]},
            {"messages": [{"role": "user", "content": "How are you?"}, {"role": "assistant", "content": "I'm doing well, thank you!"}]}
        ]
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as temp_file:
            for item in data:
                json.dump(item, temp_file)
                temp_file.write('\n')
        yield temp_file.name
        os.unlink(temp_file.name)

    @pytest.fixture
    def invalid_jsonl_file(self):
        data = [
            {"invalid": "data"},
            {"messages": [{"role": "invalid", "content": "This is not valid."}]}
        ]
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as temp_file:
            for item in data:
                json.dump(item, temp_file)
                temp_file.write('\n')
        yield temp_file.name
        os.unlink(temp_file.name)

    @pytest.fixture
    def actual_jsonl_file(self):
        data = [
            {"messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "What's the capital of France?"},
                {"role": "assistant", "content": "The capital of France is Paris."}
            ]},
            {"messages": [
                {"role": "user", "content": "Tell me a joke."},
                {"role": "assistant", "content": "Why don't scientists trust atoms? Because they make up everything!"}
            ]},
            {"messages": [
                {"role": "system", "content": "You are a math tutor."},
                {"role": "user", "content": "What's 2 + 2?"},
                {"role": "assistant", "content": "2 + 2 equals 4."}
            ]}
        ]
        
        file_path = os.path.join(os.path.dirname(__file__), 'test_data.jsonl')
        with open(file_path, 'w') as f:
            for item in data:
                json.dump(item, f)
                f.write('\n')
        
        yield file_path
        
        # Clean up the file after the test
        os.remove(file_path)

    def test_validate_openai_chat_format(self):
        preparator = DataPreparator("dummy_path.jsonl")
        valid_data = OpenAIChatFormat(messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"}
        ])
        assert preparator.validate_openai_chat_format(valid_data) == True

        invalid_data = OpenAIChatFormat(messages=[
            {"role": "invalid_role", "content": "This is invalid."},
        ])
        assert preparator.validate_openai_chat_format(invalid_data) == False

    def test_format_data_for_gemini(self):
        input_data = [OpenAIChatFormat(messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"}
        ])]
        preparator = DataPreparator("dummy_path.jsonl")
        result = preparator.format_data_for_gemini(input_data)
        assert len(result) == 1
        assert isinstance(result[0], GeminiFinetuningData)
        assert result[0].text_input == "<system>You are a helpful assistant.</system> <user>Hello!</user>"
        assert result[0].output == "<assistant>Hi there!</assistant>"

    def test_load_and_validate_data(self, valid_jsonl_file):
        preparator = DataPreparator(valid_jsonl_file)
        result = preparator.load_and_validate_data()
        print(f"Result: {result}")  # Debug print
        assert len(result) == 2
        assert all(isinstance(item, OpenAIChatFormat) for item in result)
        assert all('messages' in item for item in result)

    def test_load_and_validate_data_with_invalid_file(self, invalid_jsonl_file):
        preparator = DataPreparator(invalid_jsonl_file)
        with pytest.raises(InvalidDataFormatError):
            preparator.load_and_validate_data()

    def test_load_and_validate_data_with_nonexistent_file(self):
        preparator = DataPreparator("nonexistent_file.jsonl")
        with pytest.raises(FileNotFoundError):
            preparator.load_and_validate_data()

    def test_prepare_data_end_to_end(self, valid_jsonl_file):
        preparator = DataPreparator(valid_jsonl_file)
        result = preparator.prepare_data()
        assert len(result) == 2
        assert all(isinstance(item, GeminiFinetuningData) for item in result)
        assert all(hasattr(item, 'text_input') and hasattr(item, 'output') for item in result)

        # Check the content of the first item
        assert result[0].text_input == "<system>You are a helpful assistant.</system> <user>Hello!</user>"
        assert result[0].output == "<assistant>Hi there!</assistant>"

        # Check the content of the second item
        assert result[1].text_input == "<user>How are you?</user>"
        assert result[1].output == "<assistant>I'm doing well, thank you!</assistant>"

    def test_prepare_data_with_invalid_file(self, invalid_jsonl_file):
        preparator = DataPreparator(invalid_jsonl_file)
        with pytest.raises(InvalidDataFormatError):
            preparator.prepare_data()

    def test_prepare_data_with_actual_file(self, actual_jsonl_file):
        preparator = DataPreparator(actual_jsonl_file)
        result = preparator.prepare_data()
        
        assert len(result) == 3
        assert all(isinstance(item, GeminiFinetuningData) for item in result)
        assert all(hasattr(item, 'text_input') and hasattr(item, 'output') for item in result)
        
        # Check the content of each item
        assert result[0].text_input == "<system>You are a helpful AI assistant.</system> <user>What's the capital of France?</user>"
        assert result[0].output == "<assistant>The capital of France is Paris.</assistant>"
        
        assert result[1].text_input == "<user>Tell me a joke.</user>"
        assert result[1].output == "<assistant>Why don't scientists trust atoms? Because they make up everything!</assistant>"
        
        assert result[2].text_input == "<system>You are a math tutor.</system> <user>What's 2 + 2?</user>"
        assert result[2].output == "<assistant>2 + 2 equals 4.</assistant>"

        # Verify the file was created and contains the expected data
        with open(actual_jsonl_file, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 3
            for line in lines:
                assert json.loads(line)  # Ensure each line is valid JSON