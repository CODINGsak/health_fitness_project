import pandas as pd

class RuleBasedRecommender:
    """
    健康指標とリスクスコアに基づき、運動タイプを提案するルールベースの推奨システム
    """

    def __init__(self):
        # 운동 추천 기준을 클래스 내부에서 관리
        self.rules = {
            "high_risk": "有酸素運動",
            "metabolic_risk": "有酸素運動+筋トレ",
            "low_body_fat": "筋力トレーニング",
            "default": "ストレッチ"
        }

    def recommend(self, row):
        """
        個人の健康データに基づき、運動タイプを推奨する関数
        """

        # 1) リスクスコア基盤オススメ
        if row["risk_score"] >= 5:
            return self.rules["high_risk"]

        # 2) (glucose/LDL) チェック
        if row["glucose"] > 120 or row["ldl"] > 160:
            return self.rules["metabolic_risk"]

        # 3) 体脂肪率足りない場合
        if row["body_fat"] < 20:
            return self.rules["low_body_fat"]

        # 4) 基本オススメ
        return self.rules["default"]

    def apply_recommendation(self, df):
        """
        DataFrame全体に対して推奨結果を生成する関数
        """
        df["exercise_recommendation"] = df.apply(self.recommend, axis=1)
        return df
