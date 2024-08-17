import json
import logging
from typing import List
from .chat_message_formatters import OpenAIChatFormat
from .gemini_finetuning_data import GeminiFinetuningData
from .exceptions import InvalidDataFormatError, InvalidJSONError

class DataPreparator:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.logger = logging.getLogger(__name__)

    def validate_openai_chat_format(self, data: OpenAIChatFormat) -> bool:
        """Validate if the data follows the OpenAI chat format."""
        if not isinstance(data, OpenAIChatFormat) or "messages" not in data:
            return False
        for message in data["messages"]:
            if not isinstance(message, dict) or "role" not in message or "content" not in message:
                return False
            if message["role"] not in ["system", "user", "assistant"]:
                return False
        return True

    def load_and_validate_data(self) -> List[OpenAIChatFormat]:
        """Load training data from a JSONL file and validate its format."""
        data = []
        self.logger.info(f"Starting to load and validate data from {self.file_path}")

        try:
            with open(self.file_path, 'r') as f:
                for line_number, line in enumerate(f, 1):
                    try:
                        item = json.loads(line)
                        if 'messages' not in item:
                            raise InvalidDataFormatError(f"Missing 'messages' key in line {line_number}")
                        chat_format = OpenAIChatFormat(messages=item['messages'])
                        if not self.validate_openai_chat_format(chat_format):
                            raise InvalidDataFormatError(f"Invalid OpenAI chat format in line {line_number}")
                        data.append(chat_format)
                    except json.JSONDecodeError:
                        raise InvalidJSONError(f"Invalid JSON in line {line_number}")
        except FileNotFoundError:
            self.logger.error(f"File not found: {self.file_path}")
            raise
        except (InvalidDataFormatError, InvalidJSONError) as e:
            self.logger.error(str(e))
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error while loading file: {str(e)}")
            raise

        if not data:
            error_msg = f"No valid data found in {self.file_path}"
            self.logger.error(error_msg)
            raise InvalidDataFormatError(error_msg)

        self.logger.info(f"Successfully loaded and validated {len(data)} items from {self.file_path}")
        return data

    def format_data_for_gemini(self, data: List[OpenAIChatFormat]) -> List[GeminiFinetuningData]:
        """Format the OpenAI chat data into the required structure for Gemini finetuning."""
        return [
            GeminiFinetuningData(
                text_input=OpenAIChatFormat.format_input(item),
                output=OpenAIChatFormat.format_output(item)
            )
            for item in data
        ]
    
    def prepare_data(self) -> List[GeminiFinetuningData]:
        """Load, validate, and format data for Gemini finetuning."""
        try:
            raw_data = self.load_and_validate_data()
            formatted_data = self.format_data_for_gemini(raw_data)
            
            # Print word counts for each data point 
            for idx, data in enumerate(formatted_data):
                input_word_count = len(data.text_input.split())
                output_word_count = len(data.output.split())
                print(f"Data point {idx + 1}: Input words: {input_word_count}, Output words: {output_word_count}")
            
            return formatted_data
        except (InvalidDataFormatError, InvalidJSONError, FileNotFoundError) as e:
            raise
        except Exception as e:
            raise
