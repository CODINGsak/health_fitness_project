import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os
# sklearn:전통적인 머신러닝의 표준 도구 상자
# ---------------------------------------------------------
# 1) 데이터 로드 함수
# ---------------------------------------------------------
def load_data(path="data/processed/health_data_processed.csv"):
    """
    전처리 + feature engineering까지 완료된 CSV 파일을 불러오는 함수.
    모델 학습 단계에서는 항상 '정제된 데이터'를 사용해야 함.
    """
    return pd.read_csv(path)


# ---------------------------------------------------------
# 2) Feature / Label 분리 + 인코딩
# ---------------------------------------------------------
def prepare_features(df):
    """
    머신러닝 모델에 입력할 X(특징)와 y(라벨)를 준비하는 함수.

    - gender는 문자열(M/F)이므로 one-hot encoding 필요
    - exercise_label은 문자열이므로 LabelEncoder로 숫자 변환
    """

    # gender → one-hot encoding (문자열을 0/1 컬럼으로 변환)
    df = pd.get_dummies(df, columns=["gender"], drop_first=True)

    # y: 예측해야 하는 운동 추천 라벨
    y = df["exercise_label"]

    # X: 라벨을 제외한 모든 feature
    X = df.drop(columns=["exercise_label"])

    # 문자열 라벨을 숫자로 변환 (예: 유산소 → 0, 근력 → 1 ...)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    return X, y_encoded, le


# ---------------------------------------------------------
# 3) 모델 학습 함수
# ---------------------------------------------------------
def train_models(X, y):
    """
    KNN, Decision Tree 두 가지 모델을 학습시키고
    성능을 출력하는 함수.

    - train_test_split: 학습/테스트 데이터 분리
    - KNN: 가까운 이웃 기반 분류
    - Decision Tree: 규칙 기반 트리 모델
    """

    # 데이터 분리 (20%는 테스트용)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -------------------------
    # KNN 모델 학습
    # -------------------------
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    knn_pred = knn.predict(X_test)

    # -------------------------
    # Decision Tree 모델 학습
    # -------------------------
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X_train, y_train)
    dt_pred = dt.predict(X_test)

    # -------------------------
    # 성능 출력
    # -------------------------
    print("=== KNN 모델 성능 ===")
    print("Accuracy:", accuracy_score(y_test, knn_pred))

    print("\n=== Decision Tree 모델 성능 ===")
    print("Accuracy:", accuracy_score(y_test, dt_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, dt_pred))

    return knn, dt


# ---------------------------------------------------------
# 4) 모델 저장 함수
# ---------------------------------------------------------
def save_model(model, path="models/model.pkl"):
    """
    학습된 모델을 pickle 파일로 저장하는 함수.
    Streamlit 앱에서 불러와서 예측에 사용함.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


# ---------------------------------------------------------
# 5) 전체 파이프라인 실행 (선택)
# ---------------------------------------------------------
def run_training_pipeline():
    """
    전체 학습 파이프라인을 한 번에 실행하는 함수.
    - 데이터 로드
    - feature 준비
    - 모델 학습
    - 모델 저장
    """

    print("데이터 로드 중...")
    df = load_data()

    print("Feature 준비 중...")
    X, y, label_encoder = prepare_features(df)

    print("모델 학습 중...")
    knn_model, dt_model = train_models(X, y)

    print("모델 저장 중...")
    save_model(dt_model, "models/decision_tree.pkl")
    save_model(knn_model, "models/knn.pkl")

    print("학습 완료!")


# ---------------------------------------------------------
# 6) 직접 실행 시 동작
# ---------------------------------------------------------
if __name__ == "__main__":
    run_training_pipeline()
