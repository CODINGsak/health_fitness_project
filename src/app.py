import streamlit as st
import pandas as pd
import pickle

# ---------------------------------------------------------
# 1) 저장된 모델 불러오기
# ---------------------------------------------------------
@st.cache_resource
def load_model(path):
    """
    pickle로 저장된 머신러닝 모델을 불러오는 함수.
    Streamlit에서는 매번 다시 로드하지 않도록 cache 사용.
    """
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


# ---------------------------------------------------------
# 2) 사용자 입력 UI 구성
# ---------------------------------------------------------
def user_input_form():
    """
    Streamlit에서 사용자에게 건강 정보를 입력받는 UI.
    입력값을 DataFrame 형태로 반환하여 모델 예측에 사용.
    """

    st.header("🏋️ 건강 정보 입력")

    age = st.number_input("나이", 20, 80, 30)
    gender = st.selectbox("성별", ["M", "F"])
    height = st.number_input("키(cm)", 140, 200, 170)
    weight = st.number_input("몸무게(kg)", 40, 120, 70)
    body_fat = st.number_input("체지방률(%)", 5.0, 50.0, 20.0)
    systolic_bp = st.number_input("수축기 혈압", 90, 200, 120)
    diastolic_bp = st.number_input("이완기 혈압", 50, 130, 80)
    glucose = st.number_input("공복 혈당", 60, 200, 100)
    ldl = st.number_input("LDL 콜레스테롤", 50, 250, 120)
    hdl = st.number_input("HDL 콜레스테롤", 20, 100, 50)
    triglyceride = st.number_input("중성지방", 30, 400, 120)
    ast = st.number_input("AST", 5, 100, 25)
    alt = st.number_input("ALT", 5, 100, 25)
    steps = st.number_input("하루 걸음 수", 0, 30000, 5000)
    sleep_hours = st.number_input("수면 시간(시간)", 0.0, 12.0, 7.0)

    # BMI 계산
    height_m = height / 100
    bmi = weight / (height_m ** 2)

    # 위험 점수 계산 (feature_engineering.py와 동일한 로직)
    risk_score = 0
    if systolic_bp > 140:
        risk_score += 2
    elif systolic_bp > 130:
        risk_score += 1

    if body_fat > 30:
        risk_score += 2
    elif body_fat > 25:
        risk_score += 1

    if glucose > 120:
        risk_score += 2
    elif glucose > 110:
        risk_score += 1

    if ldl > 160:
        risk_score += 2
    elif ldl > 130:
        risk_score += 1

    # 입력값을 DataFrame으로 변환
    data = {
        "age": [age],
        "gender": [gender],
        "height": [height],
        "weight": [weight],
        "body_fat": [body_fat],
        "systolic_bp": [systolic_bp],
        "diastolic_bp": [diastolic_bp],
        "glucose": [glucose],
        "ldl": [ldl],
        "hdl": [hdl],
        "triglyceride": [triglyceride],
        "ast": [ast],
        "alt": [alt],
        "steps": [steps],
        "sleep_hours": [sleep_hours],
        "bmi": [bmi],
        "risk_score": [risk_score],
    }

    df = pd.DataFrame(data)
    return df


# ---------------------------------------------------------
# 3) 모델 예측 함수
# ---------------------------------------------------------
def predict_exercise(model, df):
    """
    모델이 예측할 수 있도록 입력 데이터를 전처리하는 함수.
    - gender는 one-hot encoding 필요
    - model_trainer.py에서 사용한 feature 형태와 동일하게 맞춰야 함
    """

    # gender(M/F) → one-hot encoding
    df = pd.get_dummies(df, columns=["gender"], drop_first=True)

    # 모델 학습 시 사용한 feature 순서와 동일하게 맞추기
    expected_columns = [
        "age", "height", "weight", "body_fat",
        "systolic_bp", "diastolic_bp", "glucose",
        "ldl", "hdl", "triglyceride", "ast", "alt",
        "steps", "sleep_hours", "bmi", "risk_score",
        "gender_M"  # one-hot encoding 결과
    ]

    # 누락된 컬럼이 있으면 0으로 채움
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[expected_columns]

    # 예측
    prediction = model.predict(df)[0]

    # 숫자 → 라벨 변환은 model_trainer에서 저장한 LabelEncoder를 불러와야 함
    return prediction


# ---------------------------------------------------------
# 4) Streamlit 메인 실행
# ---------------------------------------------------------
def main():
    st.title("🏃 건강 기반 운동 추천 시스템")

    # 모델 불러오기
    dt_model = load_model("../models/decision_tree.pkl")

    # 사용자 입력
    user_df = user_input_form()

    if st.button("운동 추천 받기"):
        pred = predict_exercise(dt_model, user_df)

        # 숫자 라벨 → 문자열 라벨 매핑
        label_map = {
            0: "근력",
            1: "스트레칭",
            2: "유산소",
            3: "유산소+근력"
        }

        st.subheader("📌 추천 운동 유형")
        st.success(label_map.get(pred, "알 수 없음"))


if __name__ == "__main__":
    main()
