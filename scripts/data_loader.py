import pandas as pd

DATA_PATH = "data/bank-additional/bank-additional/bank-additional-full.csv"
#D:\GITHUB\data\bank-additional\bank-additional\bank-additional-full.csv
# DATA_PATH = "data/bank-additional-full.csv"


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Loads the UCI Bank Marketing dataset (semicolon-separated).
    Source: S. Moro, P. Cortez, P. Rita (2014), Decision Support Systems.
    """
    df = pd.read_csv(path, sep=";")
    df.columns = [c.strip() for c in df.columns]
    return df
