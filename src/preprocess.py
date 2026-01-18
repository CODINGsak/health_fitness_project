import pandas as pd

def label_exercise(row):
    if row["systolic_bp"] > 140 or row["body_fat"] > 30:
        return "유산소"
    if row["glucose"] > 120 or row["ldl"] > 160:
        return "유산소+근력"
    if row["body_fat"] < 20:
        return "근력"
    return "스트레칭"

def preprocess_data(df):
    df["exercise_label"] = df.apply(label_exercise, axis=1)
    return df
