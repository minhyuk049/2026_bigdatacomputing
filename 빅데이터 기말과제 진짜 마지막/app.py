# 라이브러리 임포트: 웹 UI, 수치 연산, 모델 로드, 데이터 가공을 위한 핵심 패키지들
import streamlit as st  # Streamlit 웹 애플리케이션 제작
import numpy as np  # 입력값 배열 변환
import pandas as pd  # 표 형태 데이터 처리
import joblib  # 저장된 모델 및 메타데이터 로드
import matplotlib.pyplot as plt  # 성능 비교 막대그래프 출력
# 실습 코드를 긁어서 사용하였습니다

st.set_page_config(page_title="기대수명 분류 파이프라인 모델", layout="wide") # 웹 브라우저 페이지 설정

st.title("기대수명 분류 예측 및 웹 대시보드") # 대시보드 제목
st.write("가상 데이터로 학습한 Logistic Regression, k-NN, SVM 모델을 기반으로 성능 비교와 실시간 분류 예측을 수행합니다.") # 대시보드의 목적과 사용된 알고리즘 구조에 대한 간략한 설명
st.markdown("---") # 화면을 깔끔하게 분리해 주는 가로 구분선 삽입

def load_classifier_data():
    try: # joblib를 이용해 사전 학습된 모델 객체와 성능 지표가 담긴 딕셔너리 파일을 로드
        return joblib.load("life_expectancy_classifier_models.pkl")
    except FileNotFoundError:
        return None# 실패 시 빈 값 반환

payload = load_classifier_data() # 선언한 함수를 호출하여 변수에 머신러닝 데이터 패키지(payload)를 할당

if payload is None: 
    st.error("life_expectancy_classifier_models.pkl 파일이 없습니다. 먼저 코랩에서 모델 학습 셀을 실행하세요.")
    st.stop() #Streamlit 실행중단

models = payload["models"] # payload에서 학습된 최적 모델 3종 딕셔너리를 꺼냄
evaluation_summary = payload["evaluation_summary"]  # payload에서 모델별 성능 평가 결과 딕셔너리를 꺼냄
feature_names = payload["feature_names"]  # payload에서 모델 입력 특성 이름 리스트를 꺼냄
class_names = payload["class_names"]  # payload에서 숫자 라벨과 한글 클래스명 매핑 정보를 꺼냄

# 2. 성능 비교 표 출력
st.subheader("독립된 검증 데이터(Test Data) 기반 종합 성능 비교")

summary_data = []  # 빈 리스트
for model_name, info in evaluation_summary.items():  # 딕셔너리에서 모델명과 정보 딕셔너리를 하나씩 꺼냄
    summary_data.append({  # 현재 모델의 성능 정보를 하나의 행으로 만들어 리스트에 추가
        "모델명": model_name,  
        "최적 CV 구조": f"{info['best_k']}-Fold", 
        "CV Accuracy": f"{info['best_cv_score']:.2%}", 
        "Test Accuracy (정확도)": f"{info['test_accuracy']:.2%}", 
        "Test Precision (정밀도)": f"{info['test_precision']:.2%}",  
        "Test Recall (재현율)": f"{info['test_recall']:.2%}", 
        "Test F1-Score": f"{info['test_f1_score']:.4f}"  
    })  
df_summary = pd.DataFrame(summary_data)  # 리스트 형태의 성능 데이터를 Pandas DataFrame으로 변환
st.dataframe(df_summary, use_container_width=True, hide_index=True) # 성능표를 Streamlit의 인터랙티브 표 형태로 화면에 출력


# 3. Test Accuracy 막대그래프
st.subheader("📈 모델별 Test Accuracy 비교")

graph_df = pd.DataFrame({  # 막대그래프에 사용할 모델명과 테스트 정확도 데이터를 DataFrame 생성
    "Model": list(evaluation_summary.keys()),
    "Test Accuracy": [evaluation_summary[name]["test_accuracy"] for name in evaluation_summary.keys()]
})

fig, ax = plt.subplots(figsize=(8, 4)) # 그래프 크기를 8x4로 설정하여 Figure와 Axes 객체를 생성
ax.bar(graph_df["Model"], graph_df["Test Accuracy"])
ax.set_ylim(0, 1)
ax.set_ylabel("Test Accuracy")
ax.set_xlabel("Model")
ax.set_title("Model Test Accuracy Comparison")
st.pyplot(fig)
st.markdown("---")

# 4. 사이드바 입력 UI
st.sidebar.header("새로운 데이터 입력")    # 왼쪽 사이드바 영역에 입력용 헤더 텍스트 출력
adult_mortality = st.sidebar.slider("Adult Mortality",min_value=40.0,max_value=400.0,value=220.0,step=1.0)   # 사용자가 조정할 수 있는 그 슬라이더 생성
bmi = st.sidebar.slider("BMI",min_value=10.0,max_value=40.0,value=25.0,step=0.1) # 사용자가 조정할 수 있는 그 슬라이더 생성
gdp = st.sidebar.slider("GDP",min_value=300.0,max_value=70000.0,value=9000.0,step=100.0) # 사용자가 조정할 수 있는 그 슬라이더 생성
alcohol = st.sidebar.slider("Alcohol",min_value=0.0,max_value=30.0,value=5.5,step=0.1) # 사용자가 조정할 수 있는 그 슬라이더 생성
selected_model_name = st.sidebar.selectbox("실시간 예측에 사용할 모델 선택",list(models.keys()))
input_data = np.array([[adult_mortality, bmi, gdp, alcohol]], dtype=float)

# 5. 선택 모델의 실시간 예측 결과
st.subheader(
    f" 실시간 예측 판정 "
    f"[Adult Mortality: {adult_mortality:.1f}, BMI: {bmi:.1f}, GDP: {gdp:.0f}, Alcohol: {alcohol:.1f}]"
)

selected_model = models[selected_model_name]  # 선택한 모델 이름에 해당하는 학습된 파이프라인 객체 가져오기
selected_pred = int(selected_model.predict(input_data)[0])  # 선택한모델로 입력값을 예측하고 결과 라벨을 정수로 변환
selected_class_name = class_names[selected_pred]  # 예측된 숫자 라벨을 한글 클래스명으로 변환

if hasattr(selected_model.named_steps["clf"], "predict_proba"):  # 선택한 모델의 분류기가 확률 예측 기능을 제공하는지 확인
    selected_proba = selected_model.predict_proba(input_data)[0]  # 선택한 모델로 각 클래스에 대한 예측 확률을 계산
    confidence = selected_proba[selected_pred]  
    st.metric(  
        label=f"선택 모델: {selected_model_name}",  
        value=selected_class_name, 
        delta=f"예측 확률 {confidence:.1%}" 
    ) 
else:  
    st.metric( 
        label=f"선택 모델: {selected_model_name}", 
        value=selected_class_name  
    )  

st.info("왼쪽 사이드바에서 4개 변수 값을 조절하면 선택한 모델의 예측 결과가 즉시 바뀝니다.")


# 6. 모델 3종 예측 결과 카드
st.markdown("### 모델별 상세 예측 결과")

cols = st.columns(3) # 컬럼 4개 만들기

for idx, (model_name, pipeline) in enumerate(models.items()):  # 모델 딕셔너리에서 인덱스, 모델명, 파이프라인을 하나씩 꺼내 반복
    y_pred = int(pipeline.predict(input_data)[0]) # 입력값을 예측하고 결과라벨을 정수로 변환
    pred_class = class_names[y_pred]  #예측 라벨을 한글 클래스명으로 변화 ㄴ
    if hasattr(pipeline.named_steps["clf"], "predict_proba"):
        proba = pipeline.predict_proba(input_data)[0]
        proba_text = f"예측 확률 {proba[y_pred]:.1%}"
    else:
        proba_text = "확률 제공 불가"

    with cols[idx]:  # 현재 모델 결과를 idx번째 컬럼 안에 출력
        st.info(f"### {model_name}")  # 현재 컬럼 상단에 모델명을 정보 박스로 출력
        st.metric(label="판정 결과", value=pred_class, delta=proba_text)  # 현재 모델의 판정 결과와 예측 확률을 metric 형태로 출력

        cm = evaluation_summary[model_name]["confusion_matrix"]  # 현재 모델의 혼동 행렬 데이터를 평가 결과 딕셔너리에서 꺼냄
        cm_df = pd.DataFrame(cm,  index=[f"실제 {class_names[0]}", f"실제 {class_names[1]}"],  # 행 이름에 실제 클래스명을 표시
            columns=[f"예측 {class_names[0]}", f"예측 {class_names[1]}"]  # 열 이름에 예측 클래스명을 표시
        ) 

        st.markdown("**혼동 행렬 (Confusion Matrix)**")  # 혼동 행렬 표 위에 제목을 출력
        st.table(cm_df)  # 혼동 행렬을 고정된 표 형태로 출력

        st.caption(f"선택된 최적 하이퍼파라미터: {evaluation_summary[model_name]['best_params']}")  # 현재 모델의 최적 하이퍼파라미터를 작은 설명 문구로 출력
        classifier = pipeline.named_steps["clf"]  # 파이프라인 내부에서 실제 분류기 객체만 꺼냄
        if model_name == "Logistic Regression":  # 현재 모델이 Logistic Regression인지 확인
            st.markdown("---")  # 모델 정보 영역을 구분하기 위해 가로선을 출력
            st.markdown("**로지스틱 회귀 파라미터**")  # 로지스틱 회귀 계수 영역 제목을 출력
            coef_df = pd.DataFrame({  # 로지스틱 회귀 계수를 표로 만들기 위해 DataFrame을 생성
                "Feature": feature_names, 
                "Coefficient": classifier.coef_[0]  
            })  
            st.dataframe(coef_df, hide_index=True, use_container_width=True)  # 로지스틱 회귀 계수 표를 화면에 출력
            st.write(f"절편 b: `{classifier.intercept_[0]:.4f}`")  # 로지스틱 회귀의 절편 값을 소수점 네 자리로 출력

        elif model_name == "SVM":  
            st.markdown("---") 
            st.markdown("**SVM 결정 경계 정보**")  
            if hasattr(classifier, "coef_"):  # SVM이 선형 커널이라 coef_ 속성을 가지고 있는지 확
                coef_df = pd.DataFrame({  # 선형 SVM의 계수를 표로 만들기 위해 DataFrame을 생성
                    "Feature": feature_names,  # Feature 컬럼에는 입력 특성 이름
                    "Coefficient": classifier.coef_[0]  # Coefficient 컬럼에는 SVM이 학습한 선형 계수
                })  
                st.dataframe(coef_df, hide_index=True, use_container_width=True)  # SVM 계수 표를 화면에 출력
                st.write(f"절편 b: `{classifier.intercept_[0]:.4f}`")  # 선형 SVM의 절편 값을 소수점 네 자리로 출력
            else:  
                st.caption("RBF 커널이 선택된 경우 선형 계수는 직접 표시되지 않습니다.")  # 비선형 커널에서는 선형 계수를 직접 표시할 수 없다는 설명을 출력

        elif model_name == "k-NN":  # 현재 모델이 k-NN인지 확인
            st.markdown("---")  # 모델 정보 영역을 구분하기 위해 가로선을 출력
            st.markdown("**k-NN 모델 정보**")  # k-NN 모델 정보 영역 제목을 출력
            st.write(f"이웃 개수 n_neighbors: `{classifier.n_neighbors}`")  # k-NN이 사용하는 이웃 개수를 출력
            st.write(f"가중치 방식 weights: `{classifier.weights}`")  # k-NN이 사용하는 이웃 가중치 방식을 출력
