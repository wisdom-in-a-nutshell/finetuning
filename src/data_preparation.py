import pandas as pd
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

def load_data(file_path: str) -> pd.DataFrame:
    """Load training data from a CSV file."""
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except pd.errors.EmptyDataError:
        logger.error(f"Empty CSV file: {file_path}")
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data for tuning."""
    # Implement any necessary preprocessing steps
    return df

def format_data_for_tuning(df: pd.DataFrame) -> List[Dict[str, str]]:
    """Format the data into the required structure for tuning."""
    tuning_data = []
    for _, row in df.iterrows():
        example = {
            "input_text": row["input"],
            "output_text": row["output"]
        }
        tuning_data.append(example)
    return tuning_data