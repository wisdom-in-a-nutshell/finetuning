from typing import List, Dict

class Message(Dict[str, str]):
    role: str
    content: str

class OpenAIChatFormat(dict):
    messages: List[Message]

    def __init__(self, messages: List[Message]):
        super().__init__(messages=messages)

    @classmethod
    def format_input(cls, data: 'OpenAIChatFormat') -> str:
        """Format input messages (system and user) with XML tags."""
        formatted_messages = []
        for message in data['messages']:
            if message["role"] in ["system", "user"]:
                formatted_messages.append(f"<{message['role']}>{message['content']}</{message['role']}>")
        return " ".join(formatted_messages)

    @classmethod
    def format_output(cls, data: 'OpenAIChatFormat') -> str:
        """Format the assistant's message with XML tags."""
        for message in data['messages']:
            if message["role"] == "assistant":
                return f"<assistant>{message['content']}</assistant>"
        return ""  # Return empty string if no assistant message found

