import pandas as pd

def calculate_bmi(row):
    """cm＆kg BMI計算"""
    height_m = row["height"] / 100  # cm → m
    return row["weight"] / (height_m ** 2)

def calculate_risk_score(row):
    """
    リスクスコア計算
    """
    score = 0

    # 血圧
    if row["systolic_bp"] > 140:
        score += 2
    elif row["systolic_bp"] > 130:
        score += 1

    # 체지방률
    if row["body_fat"] > 30:
        score += 2
    elif row["body_fat"] > 25:
        score += 1

    # 혈당
    if row["glucose"] > 120:
        score += 2
    elif row["glucose"] > 110:
        score += 1

    # LDL
    if row["ldl"] > 160:
        score += 2
    elif row["ldl"] > 130:
        score += 1

    return score

def add_features(df):
    df["bmi"] = df["weight"] / ((df["height"] / 100) ** 2)
    df["risk_score"] = df.apply(calculate_risk_score, axis=1)
    return df
