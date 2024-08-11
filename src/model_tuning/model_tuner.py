import os
import google.generativeai as genai
from typing import List, Dict
import time
import logging
import random

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

class ModelTuner:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.creds = None
        self.setup_credentials()

    def setup_credentials(self):
        """Set up OAuth 2.0 credentials for authentication."""
        SCOPES = [
            'https://www.googleapis.com/auth/cloud-platform',
            'https://www.googleapis.com/auth/generative-language.tuning'
        ]

        # Use environment variable for client_secret.json path
        client_secret_path = os.environ.get('CLIENT_SECRET_PATH')
        if not client_secret_path:
            raise ValueError("CLIENT_SECRET_PATH environment variable is not set")
        
        self.logger.info(f"Using client_secret.json from: {client_secret_path}")

        if os.path.exists('token.json'):
            self.creds = Credentials.from_authorized_user_file('token.json', SCOPES)
        
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                self.creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    client_secret_path, SCOPES)
                self.creds = flow.run_local_server(port=0)
            
            with open('token.json', 'w') as token:
                token.write(self.creds.to_json())

        self.logger.info("OAuth 2.0 credentials set up successfully.")
        genai.configure(credentials=self.creds)

    def tune_model(self, tuning_data: List[Dict[str, str]], name: str = None):
        """Tune the Gemini model with the provided data."""
        self.logger.info("Starting model tuning process...")
        
        if name is None:
            name = f'generate-num-{random.randint(0,10000)}'
        
        # Start the tuning process
        try:
            operation = genai.create_tuned_model(
                display_name=name,
                source_model="models/gemini-1.5-flash-001-tuning",
                training_data=tuning_data,
                epoch_count=1,
                batch_size=4,
                learning_rate=0.001,
            )
            self.logger.info(f"Tuning job started. Operation name: {operation.name}")
            return operation
        except Exception as e:
            self.logger.error(f"Error starting tuning job: {str(e)}")
            raise

    def wait_for_tuning_completion(self, operation):
        """Wait for the tuning process to complete."""
        for status in operation.wait_bar():
            self.logger.info(f"Tuning status: {status}")
            time.sleep(10)

        result = operation.result()
        self.logger.info("Tuning completed successfully.")
        return result

    def get_tuned_model(self, model_name: str):
        """Get the tuned model."""
        return genai.GenerativeModel(model_name=model_name)

    def get_available_models(self):
        """Get all available models for fine-tuning."""
        self.logger.info("Fetching available models for fine-tuning...")
        try:
            models = genai.list_models()
            fine_tunable_models = [model for model in models]
            self.logger.info(f"Found {len(fine_tunable_models)} fine-tunable models.")
            return fine_tunable_models
        except Exception as e:
            self.logger.error(f"Error fetching available models: {str(e)}")
            raise