class GeminiFinetuningData:
    def __init__(self, text_input: str, output: str):
        self.text_input = text_input
        self.output = output



    @staticmethod
    def to_gemini_format(data: 'GeminiFinetuningData') -> dict:
        """
        Convert GeminiFinetuningData to the format expected by the Gemini API.
        """

        return {
            "text_input": data.text_input,
            "output": data.output
        }


