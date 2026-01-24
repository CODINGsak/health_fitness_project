import pandas as pd
from data_loader import generate_sample_health_data, save_raw_data
from preprocess import preprocess_data
from feature_engineering import add_features
import os

def run_data_pipeline():
    print("1) 샘플 데이터 생성 중...")
    df = generate_sample_health_data()
    save_raw_data(df, "../data/raw/health_data.csv")

    print("2) 전처리(라벨링) 적용 중...")
    df = preprocess_data(df)

    print("3) Feature Engineering 적용 중...")
    df = add_features(df)

    print("4) 최종 데이터 저장 중...")
    os.makedirs("../data/processed", exist_ok=True)
    df.to_csv("../data/processed/health_data_processed.csv", index=False)

    print("데이터 파이프라인 완료!")

if __name__ == "__main__":
    run_data_pipeline()
