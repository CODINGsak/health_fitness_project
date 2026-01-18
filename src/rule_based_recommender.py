import pandas as pd

class RuleBasedRecommender:
    """
    건강 지표와 위험 점수를 기반으로 운동 유형을 추천하는 규칙 기반 추천 시스템
    """

    def __init__(self):
        # 운동 추천 기준을 클래스 내부에서 관리
        self.rules = {
            "high_risk": "유산소",
            "metabolic_risk": "유산소+근력",
            "low_body_fat": "근력",
            "default": "스트레칭"
        }

    def recommend(self, row):
        """
        한 사람(row)의 건강 데이터를 기반으로 운동 유형을 추천하는 함수
        """

        # 1) 위험 점수 기반 최우선 추천
        if row["risk_score"] >= 5:
            return self.rules["high_risk"]

        # 2) 대사 위험 요소 (혈당/LDL) 체크
        if row["glucose"] > 120 or row["ldl"] > 160:
            return self.rules["metabolic_risk"]

        # 3) 체지방률이 낮은 경우 근력 운동 추천
        if row["body_fat"] < 20:
            return self.rules["low_body_fat"]

        # 4) 기본 추천
        return self.rules["default"]

    def apply_recommendation(self, df):
        """
        DataFrame 전체에 대해 추천 결과를 생성하는 함수
        """
        df["exercise_recommendation"] = df.apply(self.recommend, axis=1)
        return df
