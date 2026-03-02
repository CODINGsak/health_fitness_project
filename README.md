프로젝트 방향성
기본 건강 지표 (입력 X)
출력 Y (라벨)
추천 운동 카테고리
예: "유산소", "근력", "유산소+근력", "저강도 스트레칭"
(rule-based 라벨링 → ML 학습 방식으로 진행)
-----------------------------------------------------------------
1월18일 프로젝트 진행상황
1.폴더생성으로 프로젝특 구조 설계
data_loader.py - 200명 건강데이터를 자동생성해서 csv포맷으로 저장
preprocess.py - 건강 데이터(DataFrame)를 입력받아 각 사람에게 적합한 운동 유형을 자동으로 라벨링하는 전처리 모듈
feature_engineering.py - 데이터 기반으로 새로운 의미 있는 설계
rule_based_recommender.py - 추천 시스템의 핵심 로직
-----------------------------------------------------------------
1월24일 프로젝트 진행상황
1.model_trainer.py: 모델학습모듈
2.app.py: 실제 몸통 파이썬 파일
3.머신러닝 기초 모델링 (KNN, Decision Tree) + scikit-learn설치
4.python model_trainer.py 모델학습과정에서 에러발생
(1)학습할 csv파일이 없어서, data_pipeline.py 코드 작성
5.Streamlit 테스트실행에서 에러 발생
(1)streamlit run app.py:app.py의 모델 로드 부분 파일패스 위치 수정
LN140 : dt_model = load_model("../models/decision_tree.pkl")
6.모델 연결 + 운동 추천 로직 구현
-----------------------------------------------------------------
1월25일 작업내용
7.UX 개선 
app.py
main()st.markdown:화면출력결과 UI개선
user_input_form():BMI자동계산 UI개선
predict_exercise():risk_score계산 UI개선
main():입력값 예외처리 대응
8.프로젝트 마무리 + Git 업로드

* (첫번째,두번째 프로젝트 개선같이고민) 
** 혹시 모르니 각 프로젝트 동작확인(예외처리및 UX개선사항 확인)
***문제 없으면  requirements.txt 및 구조정리
9.AWS 배포 실습
10.README 작성 + + 포트폴리오 정리+코드이해 
11.기술관련작성 or Notion 정리+첫번째 프로젝트 코드이해(증요) // 1월31일
12.전체 리뷰 +이력서 반영+첫번째,두번째 프로젝트 코드이해(중요) 
13.면접관련해서 왜 해당기능을 사용했는지, 실무활용, 이직회사에서의 공통// 2월1일

---------------------------------------------------------------
# 🏃‍♂️ Health & Fitness Recommendation System
> **가상 건강 데이터를 활용한 머신러닝 기반 맞춤형 운동 추천 파이프라인**

## 📌 Project Overview
- **목적**: 사용자의 기본 건강 지표를 입력받아, 잠재적 건강 위험도를 계산하고 가장 적합한 운동 유형을 예측하는 ML 서비스 구축.
- **주요 기능**: 데이터 합성 및 전처리 파이프라인 구축, Risk Score(위험도 점수) 피처 엔지니어링, Decision Tree 모델 서빙 및 인터랙티브 UI.
- **개발 기간**: 2026.01

## 🛠 Tech Stack
- **Language**: Python 3.x
- **Machine Learning**: Scikit-learn (Decision Tree, KNN)
- **Data Engineering**: Pandas, Numpy
- **Frontend / Serving**: Streamlit

## ⚙️ Key Challenges & Solutions (Troubleshooting)

### 1. 도메인 기반 파생 변수 생성 (Feature Engineering)
- **Issue**: 단순 신체/혈액 데이터만으로는 ML 모델이 고위험군 사용자를 명확히 분류하기 어려움.
- **Solution**: 의학적 기준에 따라 혈압, 혈당, 체지방률 등에 가중치를 부여한 `risk_score`를 새롭게 설계. Vectorization 연산을 통해 데이터 처리 성능을 최적화하고 모델의 분류 정확도 향상.

### 2. 알고리즘 스케일 민감도 한계 극복
- **Issue**: 초기 KNN 모델 도입 시, '키(170)'와 '위험도 점수(5)' 간의 스케일 차이로 인해 거리 계산(Distance)에 왜곡 발생.
- **Solution**: 스케일에 영향을 받지 않는 규칙 기반 분할 알고리즘인 **Decision Tree**로 모델을 교체하여 문제를 해결하고, 향후 데이터 정규화(StandardScaler) 파이프라인 도입 필요성 확인.

### 3. Model Serving 및 데이터 무결성 확보
- **Issue**: UI 입력 변수명과 모델 학습 변수명 간의 스키마 불일치(Schema Mismatch)로 인한 추론 오류 가능성.
- **Solution**: 추론부(`predict_exercise`) 진입 시 동적 매핑 로직을 추가하여 데이터 무결성을 확보하고, 나이/혈압 등 논리적 오류를 사전에 차단하는 방어적 로직(Validation)을 프론트엔드 단에 구현.

## 📂 Project Structure
- `data_loader.py`: 가상 회원 데이터 합성 및 CSV 저장
- `feature_engineering.py`: BMI, Risk Score 파생 변수 생성 로직
- `preprocess.py`: Rule-based 기반 학습용 Target Labeling
- `model_trainer.py`: 모델 학습, 검증(Train-Test Split) 및 객체 직렬화(Pickle)
- `app.py`: @st.cache_resource 기반 모델 서빙 및 대시보드 UI