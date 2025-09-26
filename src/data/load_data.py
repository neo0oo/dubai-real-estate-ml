import pandas as pd
from pathlib import Path

def load_raw_data(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)

if __name__ == "__main__":
    test_path = Path("data/raw/dubai_properties.csv")
    df = load_raw_data(test_path)
    print(df.head())