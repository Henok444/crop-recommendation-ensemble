import pandas as pd 
from pathlib import Path
import os 
from dotenv import load_dotenv

load_dotenv()

data_dir = Path(os.getenv("data_dir"))


def load_data():
    file_path = data_dir / "Crop_recommendation.csv"
    df = pd.DataFrame(pd.read_csv(file_path))
    return df 
