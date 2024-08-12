import logging
import os

from google import generativeai as genai
from google.ai import generativelanguage as glm
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow


class BaseModelHandler:
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

    def get_tuned_model_status(self, model_name: str):
        """Get the status of a tuned model."""
        self.logger.info(f"Checking status of tuned model: {model_name}")
        try:
            # Attempt to get the model
            model = genai.get_model(model_name)

            # Check the model's state
            if model.state == glm.TunedModel.State.ACTIVE:
                status = "Ready for use"
            elif model.state == glm.TunedModel.State.CREATING:
                status = "Still being created"
            elif model.state == glm.TunedModel.State.FAILED:
                status = "Creation failed"
            elif model.state == glm.TunedModel.State.STATE_UNSPECIFIED:
                status = "State unspecified"
            else:
                status = f"Unknown state: {model.state.name}"
            self.logger.info(f"Model status: {status}")
            return status
        except Exception as e:
            self.logger.error(f"Error checking model status: {str(e)}")
            raise

    def get_tuned_model(self, model_name: str):
        """Get the tuned model."""
        return genai.GenerativeModel(model_name=model_name)

    def get_tuned_models(self):
        """Get all tuned models available to the user."""
        self.logger.info("Fetching all tuned models...")
        try:
            tuned_models = genai.list_tuned_models()
            return tuned_models  # This is a TunedModelsIterable
        except Exception as e:
            self.logger.error(f"Error fetching tuned models: {str(e)}")
            raise