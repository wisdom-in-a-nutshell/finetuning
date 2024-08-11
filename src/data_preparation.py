import json
import logging
from typing import List, Dict, TypedDict

class Message(TypedDict):
    role: str
    content: str

class OpenAIChatFormat(TypedDict):
    messages: List[Message]

class GeminiFinetuningData(TypedDict):
    input_text: str
    output_text: str

class InvalidDataFormatError(Exception):
    """Raised when the data format is invalid."""
    pass

class InvalidJSONError(Exception):
    """Raised when the JSON format is invalid."""
    pass

def validate_openai_chat_format(data: OpenAIChatFormat) -> bool:
    """Validate if the data follows the OpenAI chat format."""
    if not isinstance(data, dict) or "messages" not in data:
        return False
    for message in data["messages"]:
        if not isinstance(message, dict) or "role" not in message or "content" not in message:
            return False
        if message["role"] not in ["system", "user", "assistant"]:
            return False
    return True

def load_and_validate_data(file_path: str) -> List[OpenAIChatFormat]:
    """Load training data from a JSONL file and validate its format."""
    data = []
    logging.info(f"Starting to load and validate data from {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            for line_number, line in enumerate(f, 1):
                try:
                    item = json.loads(line)
                    if not validate_openai_chat_format(item):
                        raise InvalidDataFormatError(f"Invalid OpenAI chat format in line {line_number}")
                    data.append(item)
                except json.JSONDecodeError:
                    raise InvalidJSONError(f"Invalid JSON in line {line_number}")
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise
    except (InvalidDataFormatError, InvalidJSONError) as e:
        logging.error(str(e))
        raise
    except Exception as e:
        logging.error(f"Unexpected error while loading file: {str(e)}")
        raise

    if not data:
        error_msg = f"No valid data found in {file_path}"
        logging.error(error_msg)
        raise InvalidDataFormatError(error_msg)

    logging.info(f"Successfully loaded and validated {len(data)} items from {file_path}")
    return data

def format_data_for_gemini(data: List[OpenAIChatFormat]) -> List[GeminiFinetuningData]:
    """Format the OpenAI chat data into the required structure for Gemini finetuning."""
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

def prepare_data_for_gemini(file_path: str) -> List[GeminiFinetuningData]:
    """Load, validate, and format data for Gemini finetuning."""
    try:
        raw_data = load_and_validate_data(file_path)
        return format_data_for_gemini(raw_data)
    except (InvalidDataFormatError, InvalidJSONError, FileNotFoundError) as e:
        logging.error(f"Error preparing data for Gemini: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error preparing data for Gemini: {str(e)}")
        raise