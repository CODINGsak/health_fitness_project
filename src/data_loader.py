import pandas as pd
import numpy as np
import os

#200명 건강데이터 샘플로 만들기
def generate_sample_health_data(n=200):
    #seed를 고정하면 항상 같은 값이 생성됨
    np.random.seed(42)

    data = {
        "age": np.random.randint(20, 70, n),
        "gender": np.random.choice(["M", "F"], n),
        "height": np.random.randint(150, 190, n),
        "weight": np.random.randint(50, 100, n),
        "body_fat": np.random.uniform(10, 35, n),
        "systolic_bp": np.random.randint(100, 160, n),
        "diastolic_bp": np.random.randint(60, 100, n),
        "glucose": np.random.randint(70, 150, n),
        "ldl": np.random.randint(80, 200, n),
        "hdl": np.random.randint(30, 80, n),
        "triglyceride": np.random.randint(50, 250, n),
        "ast": np.random.randint(10, 60, n),
        "alt": np.random.randint(10, 60, n),
        "steps": np.random.randint(1000, 15000, n),
        "sleep_hours": np.random.uniform(4, 9, n)
    }

    df = pd.DataFrame(data)
    return df

def save_raw_data(df, path="data/raw/health_data.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
