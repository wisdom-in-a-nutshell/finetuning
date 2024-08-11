from typing import List, TypedDict

class Message(TypedDict):
    role: str
    content: str

class OpenAIChatFormat(TypedDict):
    messages: List[Message]

class GeminiFinetuningData(TypedDict):
    text_input: str
    output: str