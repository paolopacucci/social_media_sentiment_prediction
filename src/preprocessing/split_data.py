from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

processed_data_path = Path("data/processed/sentiment_preprocessed.csv")
split_dir = Path("data/split")
train_path = split_dir / "train_df.csv"
test_path = split_dir / "test_df.csv"

def load_preprocessed_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = df["text"]
    y = df["label"]
    RANDOM_SEED = 42
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
    
    train_df = pd.DataFrame({"text": X_train, "label": y_train})
    test_df = pd.DataFrame({"text": X_test, "label": y_test })
    
    return train_df, test_df

def save_split_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

def main():
    df = load_preprocessed_data(processed_data_path)
    train_df, test_df = split_data(df)
    save_split_data(train_df, test_df)

if __name__ == "__main__":
    main()

        
