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
        self.model = None
        self.tuning_job = None
        self.creds = None
        self.setup_credentials()
        self.setup_model()

    def setup_credentials(self):
        """Set up OAuth 2.0 credentials for authentication."""
        SCOPES = ['https://www.googleapis.com/auth/cloud-platform',
                  'https://www.googleapis.com/auth/generative-language.retriever']

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

    def setup_model(self):
        """Set up the Gemini model for tuning."""
        genai.configure(credentials=self.creds)
        all_models = genai.list_models()
        tunable_models = [m for m in all_models if "createTunedModel" in m.supported_generation_methods]
        if not tunable_models:
            raise ValueError("No models supporting fine-tuning found")
        self.model = tunable_models[0]
        self.logger.info(f"Selected model for tuning: {self.model.name}")

    def tune_model(self, tuning_data: List[Dict[str, str]], name: str = None):
        """Tune the Gemini model with the provided data."""
        self.logger.info("Starting model tuning process...")
        
        if name is None:
            name = f'generate-num-{random.randint(0,10000)}'
        
        # Start the tuning process
        try:
            self.tuning_job = genai.create_tuned_model(
                source_model=self.model.name,
                training_data=tuning_data,
                id=name,
                epoch_count=100,
                batch_size=4,
                learning_rate=0.001,
            )
            self.logger.info(f"Tuning job started. Model name: {self.tuning_job.name}")
            return self.tuning_job
        except Exception as e:
            self.logger.error(f"Error starting tuning job: {str(e)}")
            raise

    def get_tuned_model_status(self):
        """Check the status of the tuning job."""
        if not self.tuning_job:
            raise ValueError("No tuning job has been started")
        model = genai.get_tuned_model(self.tuning_job.name)
        return model.state


    def wait_for_tuning_completion(self):
        """Wait for the tuning process to complete."""
        if not self.tuning_job:
            raise ValueError("No tuning job has been started")
        while True:
            status = self.get_tuned_model_status()
            self.logger.info(f"Tuning status: {status}")
            if status == "ACTIVE":
                self.logger.info("Tuning completed successfully.")
                return genai.get_tuned_model(self.tuning_job.name)
            elif status in ["FAILED", "CANCELLED"]:
                raise Exception(f"Tuning failed with status: {status}")
            time.sleep(60)  # Check status every minute