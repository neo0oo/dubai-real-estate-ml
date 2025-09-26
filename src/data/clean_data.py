from pathlib import Path
import pandas as pd
from .load_data import load_raw_data

RAW_PATH = Path("data/raw/dubai_properties.csv")
OUT_PATH = Path("data/processed/cleaned_dubai_properties_clean.csv")

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates()
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    return df

if __name__ == "__main__":
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw data file not found at {RAW_PATH}. Put CSV file there.")
    df = load_raw_data(RAW_PATH)
    df_clean = clean(df)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(OUT_PATH, index=False)
    print(f"Cleaned data saved to {OUT_PATH}")