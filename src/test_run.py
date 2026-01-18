import pandas as pd

from preprocess import preprocess_data
from feature_engineering import add_features
from rule_based_recommender import RuleBasedRecommender

# 1) 테스트용 더미 데이터 생성
data = {
    "height": [170, 165, 180],
    "weight": [70, 85, 60],
    "systolic_bp": [150, 135, 120],
    "body_fat": [28, 32, 18],
    "glucose": [110, 130, 95],
    "ldl": [120, 170, 100]
}

df = pd.DataFrame(data)

# 2) 전처리 라벨링
df = preprocess_data(df)

# 3) feature engineering
df = add_features(df)

# 4) 규칙 기반 추천
recommender = RuleBasedRecommender()
df = recommender.apply_recommendation(df)

# 5) 결과 출력
print(df)
