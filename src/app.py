import streamlit as st
import pandas as pd
import pickle

# ---------------------------------------------------------
# 1) 保存されたモデル呼出
# ---------------------------------------------------------
@st.cache_resource
def load_model(path):
    """
    pickle形式で保存された学習済みモデルをロードする関数
    Streamlitでの再ロードを避けるため、キャッシュ機能（cache_resource）を使用
    """
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


# ---------------------------------------------------------
# 2) ユーザー入力
# ---------------------------------------------------------
def user_input_form():
    st.header("🏋️ 健康情報の入力")

    age = st.number_input("年齢", min_value=1, max_value=120, value=30)
    gender = st.selectbox("性別", ["M", "F"])
    height = st.number_input("身長(cm)", min_value=100, max_value=250, value=170)
    weight = st.number_input("体重(kg)", min_value=30, max_value=200, value=70)

    #BMI
    bmi = weight / ((height / 100) ** 2)
    st.markdown(f"**📌 自動算出されたBMI:** `{bmi:.2f}`")

    body_fat = st.number_input("体脂肪率(%)", min_value=1.0, max_value=60.0, value=20.0)
    sbp = st.number_input("収縮期血圧", min_value=80, max_value=200, value=120)
    dbp = st.number_input("拡張期血圧", min_value=50, max_value=130, value=80)
    glucose = st.number_input("空腹時血糖値", min_value=60, max_value=200, value=100)
    ldl = st.number_input("LDL", min_value=50, max_value=300, value=120)
    hdl = st.number_input("HDL", min_value=20, max_value=100, value=50)
    tg = st.number_input("中性脂肪", min_value=30, max_value=500, value=120)
    ast = st.number_input("AST", min_value=5, max_value=200, value=25)
    alt = st.number_input("ALT", min_value=5, max_value=200, value=25)
    steps = st.number_input("1日の歩数", min_value=0, max_value=50000, value=5000)
    sleep = st.number_input("睡眠時間", min_value=0.0, max_value=24.0, value=7.0)

    # 🔥 risk_score
    risk_score = (
        (sbp - 120) * 0.3 +
        (dbp - 80) * 0.2 +
        (glucose - 100) * 0.3 +
        (body_fat - 20) * 0.2
    )

    st.markdown(f"**📊 リスクスコア(Risk Score):** `{risk_score:.2f}`")

    # DataFrame
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
        "risk_score": [risk_score]
    })

    return df

# ---------------------------------------------------------
# 3) モデルによる予測
# ---------------------------------------------------------
def predict_exercise(model, df):
    """
    モデル推論用に入力データを前処理する関数
    """
    df = df.rename(columns={
        "sbp": "systolic_bp",
        "dbp": "diastolic_bp",
        "sleep": "sleep_hours",
        "tg": "triglyceride"
    })

    # risk_scoreはdf内に含まれていると想定
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
# 4) Streamlit 実行
# ---------------------------------------------------------
def main():
    st.title("🏃 ヘルスケアデータに基づく運動推奨システム")

    dt_model = load_model("../models/decision_tree.pkl")
    user_df = user_input_form()

    if st.button("おすすめの運動を診断する"):
        # 🔍 입력값 검증
        if user_df["height"].iloc[0] <= 0:
            st.error("❗ 身長は0より大きい値を入力してください。")
            return

        if user_df["weight"].iloc[0] <= 0:
            st.error("❗ 体重は0より大きい値を入力してください。")
            return

        if user_df["sleep"].iloc[0] > 24:
            st.error("❗ 睡眠時間は24時間を超えることはできません。")
            return

        if user_df["sbp"].iloc[0] < user_df["dbp"].iloc[0]:
            st.error("❗ 収縮期血圧は拡張期血圧より高い必要があります。")
            return

        # 正常な入力値の場合、予測を実行
        pred, risk_score = predict_exercise(dt_model, user_df)

        # 数値ラベルから文字列ラベルへのマッピング
        label_map = {
            0: "筋力トレーニング",
            1: "ストレッチ",
            2: "有酸素運動",
            3: "有酸素運動+筋トレ"
        }

        exercise_type = label_map.get(pred, "不明")

        # 結果 UI
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
                <h3 style="color: #1a73e8; margin-bottom: 10px;">🏋️ 推奨される運動タイプ</h3>
                <p style="font-size: 22px; font-weight: bold; color: #333;">
                    {exercise_type}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # 🔥 risk_score visualize
        st.subheader("📊 リスクスコア (Risk Score)")
        st.write(f"**現在のリスク度:** {risk_score:.2f}")

        # プログレスバーは0〜100基準のため、
        # risk_scoreに適したスケーリングを適用（0〜20の範囲を0〜100に変換）
        scaled_score = min(max(risk_score * 5, 0), 100)
        st.progress(int(scaled_score))

        if risk_score < 5:
            st.success("🟢 良好")
        elif risk_score < 15:
            st.warning("🟡 注意")
        else:
            st.error("🔴 要管理")

if __name__ == "__main__":
    main()
