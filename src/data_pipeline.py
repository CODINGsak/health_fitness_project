import pandas as pd
from data_loader import generate_sample_health_data, save_raw_data
from preprocess import preprocess_data
from feature_engineering import add_features
import os

def run_data_pipeline():
    print("1) サンプルデータを作成中...")
    df = generate_sample_health_data()
    save_raw_data(df, "../data/raw/health_data.csv")

    print("2) 前処理（ラベリング）を適用中...")
    df = preprocess_data(df)

    print("3) Feature Engineering適用中...")
    df = add_features(df)

    print("4) 最終データの保存中...")
    os.makedirs("../data/processed", exist_ok=True)
    df.to_csv("../data/processed/health_data_processed.csv", index=False)

    print("データパイプライン処理完了!")

if __name__ == "__main__":
    run_data_pipeline()
