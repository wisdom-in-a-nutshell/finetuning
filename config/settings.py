import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = "gemini-1.5-pro"
TUNED_MODEL_NAME = "my-tuned-gemini-model"

# Add more configuration options as needed
MAX_TOKENS = 1024
TEMPERATURE = 0.7