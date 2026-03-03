import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os
# ---------------------------------------------------------
# 1) data load
# ---------------------------------------------------------
def load_data(path="data/processed/health_data_processed.csv"):
    """
    前処理と特徴量エンジニアリングが完了したCSVファイルをロードする関数
    モデルの学習フェーズでは常に「精製されたデータ」を使用する必要がある
    """
    return pd.read_csv(path)


# ---------------------------------------------------------
# 2) Feature / Label
# ---------------------------------------------------------
def prepare_features(df):
    """
    機械学習モデルに入力する特徴量(X)とラベル(y)を準備する関数
    - genderは文字列(M/F)のため、One-Hotエンコーディングが必要
    - exercise_labelは文字列のため、LabelEncoderで数値に変換
    """

    # gender → one-hot encoding
    df = pd.get_dummies(df, columns=["gender"], drop_first=True)

    # y: オススメ運動ラベル
    y = df["exercise_label"]

    # X: ラベルを除外した全ての feature
    X = df.drop(columns=["exercise_label"])

    # 文字列ラベルを数字に変換
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    return X, y_encoded, le


# ---------------------------------------------------------
# 3) モデル学習
# ---------------------------------------------------------
def train_models(X, y):
    """
    KNNと意思決定木の2つのモデルを学習させ、性能を出力する関数
    - train_test_split: 学習用データとテスト用データの分割
    - KNN: 近傍法による分類
    - Decision Tree: ルールベースの木構造モデル
    """

    # データの分割（20%をテスト用として使用）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -------------------------
    # KNN モデル学習
    # -------------------------
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    knn_pred = knn.predict(X_test)

    # -------------------------
    # Decision Tree モデル学習
    # -------------------------
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X_train, y_train)
    dt_pred = dt.predict(X_test)

    # -------------------------
    # 성능 출력
    # -------------------------
    print("=== KNN モデル性能 ===")
    print("Accuracy:", accuracy_score(y_test, knn_pred))

    print("\n=== Decision Tree モデル性能 ===")
    print("Accuracy:", accuracy_score(y_test, dt_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, dt_pred))

    return knn, dt


# ---------------------------------------------------------
# 4) モデル保存
# ---------------------------------------------------------
def save_model(model, path="models/model.pkl"):
    """
    学習済みモデルをpickleファイルとして保存する関数
    Streamlit アプリでロードし、推論に使用
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


# ---------------------------------------------------------
# 5) 全体パイプライン実行
# ---------------------------------------------------------
def run_training_pipeline():

    print("データロード中...")
    df = load_data()

    print("Feature 準備中...")
    X, y, label_encoder = prepare_features(df)

    print("モデル学習中...")
    knn_model, dt_model = train_models(X, y)

    print("モデル保存中...")
    save_model(dt_model, "models/decision_tree.pkl")
    save_model(knn_model, "models/knn.pkl")

    print("学習完了!")


# ---------------------------------------------------------
# 6) 直接実行
# ---------------------------------------------------------
if __name__ == "__main__":
    run_training_pipeline()
