import pandas as pd
import numpy as np

# ==========================================
# data_loader.py 
# ==========================================
def load_tsv(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath, sep="\t", dtype=str, keep_default_na=False)

def save_predictions(df: pd.DataFrame, output_path: str) -> None:
    df.fillna("", inplace=True)
    df.to_csv(output_path, sep="\t", index=False)