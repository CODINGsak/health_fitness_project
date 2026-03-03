import pandas as pd

def label_exercise(row):
    if row["systolic_bp"] > 140 or row["body_fat"] > 30:
        return "有酸素運動"
    if row["glucose"] > 120 or row["ldl"] > 160:
        return "有酸素運動+筋トレ"
    if row["body_fat"] < 20:
        return "筋力トレーニング"
    return "ストレッチ"

def preprocess_data(df):
    df["exercise_label"] = df.apply(label_exercise, axis=1)
    return df
