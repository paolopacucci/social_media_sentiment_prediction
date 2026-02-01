from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

processed_data_path = Path("data/processed/sentiment_preprocessed.csv")
split_dir = Path("data/split")

train_path = split_dir / "train.csv"
val_path = split_dir / "val.csv"
test_path = split_dir / "test.csv"

RANDOM_SEED = 42
TEST_SIZE = 0.20  # 20% totale
VAL_SIZE = 0.10   # 10% totale

def load_preprocessed_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def split_data(df: pd.DataFrame):
    train_val_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=df["label"]
    )

    val_size_adjusted = VAL_SIZE / (1 - TEST_SIZE)

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size_adjusted,
        random_state=RANDOM_SEED,
        stratify=train_val_df["label"]
    )

    return train_df, val_df, test_df

def save_split_data(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

def dist(df: pd.DataFrame) -> dict:
    return df["label"].value_counts(normalize=True).round(3).to_dict()

def main():
    df = load_preprocessed_data(processed_data_path)
    train_df, val_df, test_df = split_data(df)
    save_split_data(train_df, val_df, test_df)

    print("OK split salvato in data/split/")
    print("train rows:", len(train_df), "dist:", dist(train_df))
    print("val rows  :", len(val_df),   "dist:", dist(val_df))
    print("test rows :", len(test_df),  "dist:", dist(test_df))

if __name__ == "__main__":
    main()
