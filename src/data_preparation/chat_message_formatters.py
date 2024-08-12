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
        messages = {msg["role"]: msg["content"] for msg in data['messages']}
        
        if "user" not in messages:
            raise ValueError("User message is required but missing.")
        
        formatted_messages = [f"<user>{messages['user']}</user>"]
        
        if "system" in messages:
            formatted_messages.insert(0, f"<system>{messages['system']}</system>")
        
        return " ".join(formatted_messages)

    @classmethod
    def format_output(cls, data: 'OpenAIChatFormat') -> str:
        """Format the assistant's message with XML tags."""
        messages = {msg["role"]: msg["content"] for msg in data['messages']}
        
        if "assistant" not in messages:
            raise ValueError("Assistant message is required but missing.")
        
        return f"<assistant>{messages['assistant']}</assistant>"