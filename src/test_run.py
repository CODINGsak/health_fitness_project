import pandas as pd

from preprocess import preprocess_data
from feature_engineering import add_features
from rule_based_recommender import RuleBasedRecommender

# 1) テスト用データ
data = {
    "height": [170, 165, 180],
    "weight": [70, 85, 60],
    "systolic_bp": [150, 135, 120],
    "body_fat": [28, 32, 18],
    "glucose": [110, 130, 95],
    "ldl": [120, 170, 100]
}

df = pd.DataFrame(data)

# 2) 全処理ラベリング
df = preprocess_data(df)

# 3) feature engineering
df = add_features(df)

# 4) オススメ
recommender = RuleBasedRecommender()
df = recommender.apply_recommendation(df)

# 5) 結果出力
print(df)
