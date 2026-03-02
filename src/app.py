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
    st.header("🏋️ 건강 정보 입력")

    age = st.number_input("나이", min_value=1, max_value=120, value=30)
    gender = st.selectbox("성별", ["M", "F"])
    height = st.number_input("키(cm)", min_value=100, max_value=250, value=170)
    weight = st.number_input("몸무게(kg)", min_value=30, max_value=200, value=70)

    # 🔥 BMI 자동 계산
    bmi = weight / ((height / 100) ** 2)
    st.markdown(f"**📌 자동 계산된 BMI:** `{bmi:.2f}`")

    body_fat = st.number_input("체지방률(%)", min_value=1.0, max_value=60.0, value=20.0)
    sbp = st.number_input("수축기 혈압", min_value=80, max_value=200, value=120)
    dbp = st.number_input("이완기 혈압", min_value=50, max_value=130, value=80)
    glucose = st.number_input("공복 혈당", min_value=60, max_value=200, value=100)
    ldl = st.number_input("LDL 콜레스테롤", min_value=50, max_value=300, value=120)
    hdl = st.number_input("HDL 콜레스테롤", min_value=20, max_value=100, value=50)
    tg = st.number_input("중성지방", min_value=30, max_value=500, value=120)
    ast = st.number_input("AST", min_value=5, max_value=200, value=25)
    alt = st.number_input("ALT", min_value=5, max_value=200, value=25)
    steps = st.number_input("하루 걸음 수", min_value=0, max_value=50000, value=5000)
    sleep = st.number_input("수면 시간(시간)", min_value=0.0, max_value=24.0, value=7.0)

    # 🔥 risk_score 자동 계산
    risk_score = (
        (sbp - 120) * 0.3 +
        (dbp - 80) * 0.2 +
        (glucose - 100) * 0.3 +
        (body_fat - 20) * 0.2
    )

    st.markdown(f"**📊 자동 계산된 위험도 점수(Risk Score):** `{risk_score:.2f}`")

    # DataFrame 생성
    df = pd.DataFrame({
        "age": [age],
        "gender": [gender],
        "height": [height],
        "weight": [weight],
        "bmi": [bmi],
        "body_fat": [body_fat],
        "sbp": [sbp],
        "dbp": [dbp],
        "glucose": [glucose],
        "ldl": [ldl],
        "hdl": [hdl],
        "tg": [tg],
        "ast": [ast],
        "alt": [alt],
        "steps": [steps],
        "sleep": [sleep],
        "risk_score": [risk_score]   # 🔥 추가됨
    })

    return df

# ---------------------------------------------------------
# 3) 모델 예측 함수
# ---------------------------------------------------------
def predict_exercise(model, df):
    """
    모델이 예측할 수 있도록 입력 데이터를 전처리하는 함수.
    """
    df = df.rename(columns={
        "sbp": "systolic_bp",
        "dbp": "diastolic_bp",
        "sleep": "sleep_hours",
        "tg": "triglyceride" # 중성지방도 이름이 다를 수 있으니 확인!
    })

    # risk_score는 df 안에 이미 들어 있다고 가정
    risk_score = df["risk_score"].iloc[0]

    # gender(M/F) → one-hot encoding
    df = pd.get_dummies(df, columns=["gender"], drop_first=True)

    expected_columns = [
        "age", "height", "weight", "body_fat",
        "systolic_bp", "diastolic_bp", "glucose",
        "ldl", "hdl", "triglyceride", "ast", "alt",
        "steps", "sleep_hours", "bmi", "risk_score",
        "gender_M"
    ]

    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[expected_columns]

    pred = model.predict(df)[0]

    return pred, risk_score

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
        # 🔍 입력값 검증
        if user_df["height"].iloc[0] <= 0:
            st.error("❗ 키는 0보다 커야 합니다.")
            return

        if user_df["weight"].iloc[0] <= 0:
            st.error("❗ 몸무게는 0보다 커야 합니다.")
            return

        if user_df["sleep"].iloc[0] > 24:
            st.error("❗ 수면 시간은 24시간을 넘을 수 없습니다.")
            return

        if user_df["sbp"].iloc[0] < user_df["dbp"].iloc[0]:
            st.error("❗ 수축기 혈압은 이완기 혈압보다 커야 합니다.")
            return

        # 🔥 정상 입력이면 예측 실행
        pred, risk_score = predict_exercise(dt_model, user_df)

        # 숫자 라벨 → 문자열 라벨 매핑
        label_map = {
            0: "근력",
            1: "스트레칭",
            2: "유산소",
            3: "유산소+근력"
        }

        exercise_type = label_map.get(pred, "알 수 없음")

        # 🔥 추천 결과 카드 UI
        st.markdown(
            f"""
            <div style="
                background-color: #f0f8ff;
                padding: 20px;
                border-radius: 12px;
                border: 1px solid #d0e7ff;
                box-shadow: 0 4px 8px rgba(0,0,0,0.05);
                margin-top: 20px;
            ">
                <h3 style="color: #1a73e8; margin-bottom: 10px;">🏋️ 추천 운동 유형</h3>
                <p style="font-size: 22px; font-weight: bold; color: #333;">
                    {exercise_type}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # 🔥 risk_score 시각화
        st.subheader("📊 건강 위험도 점수 (Risk Score)")
        st.write(f"**현재 위험도:** {risk_score:.2f}")

        # progress bar는 0~100 기준이므로 risk_score를 그대로 사용하기 어렵기 때문에
        # 적당한 스케일링을 적용 (0~20 범위를 0~100으로 변환)
        scaled_score = min(max(risk_score * 5, 0), 100)
        st.progress(int(scaled_score))

        # 상태 메시지 (직관적인 기준)
        if risk_score < 5:
            st.success("🟢 건강 상태 양호")
        elif risk_score < 15:
            st.warning("🟡 주의 필요")
        else:
            st.error("🔴 건강 관리가 필요합니다")

if __name__ == "__main__":
    main()
