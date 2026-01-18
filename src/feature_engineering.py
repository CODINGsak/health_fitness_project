import pandas as pd

def calculate_bmi(row):
    """키(cm)와 체중(kg)을 이용해 BMI를 계산하는 함수"""
    height_m = row["height"] / 100  # cm → m 변환
    return row["weight"] / (height_m ** 2)

def calculate_risk_score(row):
    """
    건강 지표를 기반으로 위험 점수를 계산하는 함수.
    점수는 가중치 기반으로 구성.
    """
    score = 0

    # 혈압
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
    """BMI, 위험 점수 등 새로운 feature를 추가하는 메인 함수"""
    df["bmi"] = df.apply(calculate_bmi, axis=1)
    df["risk_score"] = df.apply(calculate_risk_score, axis=1)
    return df
