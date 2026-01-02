import streamlit as st
import joblib
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go

# 페이지 설정
st.set_page_config(
    page_title="자동차 연비 예측 앱",
    page_icon="🚗",
    layout="wide"
)

# 제목
st.title("🚗 자동차 연비 예측 애플리케이션")
st.markdown("---")

# 모델 로드(캐싱)
@st.cache_resource
def load_model() :
    model = joblib.load("./mpg_model.joblib")
    return model

# 모델 정보 정보 로드(캐싱)
@st.cache_data
def load_model_info() :
    with open('./model_info.json', 'r') as f :
        return json.load(f)

# 모델 및 정보 불러오기
model = load_model()
model_info = load_model_info()

# 사이드바
st.sidebar.header("⚙️ 설정")

# 모델 정보 표시
st.sidebar.subheader("📊 모델 성능")

if model_info :
    # 모델 정보가 불러온 상태라면 모델 성능 표시
    st.sidebar.metric("R² score", f"{model_info['r2_score'] : .3f}")
    st.sidebar.metric("RMSE", f"{model_info['rmse']: .2f}")
    st.sidebar.metric('MSE', f"{model_info['mse'] :.2f}")

# 메인 영역
tab1, tab2, tab3 = st.tabs(["🔮 예측", "📈 모델 정보", "📊 데이터 분석"])

# 탭 1: 예측
with tab1:
    st.header("연비 예측")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # 입력 폼
        st.subheader("입력 정보")
        weight = st.number_input(
            "자동차 무게 (lbs)",
            min_value=1000,
            max_value=6000,
            value=3000,
            step=100,
            help="예측할 자동차의 무게를 입력하세요"
        )
        
        # 예측 버튼
        if st.button("🔮 연비 예측하기", type="primary", use_container_width=True):
            # 예측 수행
            input_data = np.array([[weight]]) # 무게를 2차원 배열로 변환
            # 예측 결과
            predicted_mpg = model.predict(input_data)[0]
            
            # 세션 상태에 저장
            if 'predictions' not in st.session_state : 
                st.session_state.predictions = [] 
                # 예측 결과가 없다면 빈 리스트로 초기화
            
            # 세션에 입력한 무게와 예측 결과 추가
            st.session_state.predictions.append({
                'weight' : weight, 
                'predicted_mpg' : predicted_mpg
            })
            
    with col2:
        # 예측 결과 표시
        if st.session_state.get('predictions') : 
            # 세션이 비어있지 않은 경우 결과 표시
            latest = st.session_state.predictions[-1] # 가장 최근 예측 결과
            st.subheader('예측결과')
            st.metric(
                label = "예측 연비",
                value = f"{latest['predicted_mpg'] : .2f}", 
                delta = f"{latest['predicted_mpg'] - 23.5 : .2f}"
            )
            st.caption("단위 : mpg(miles per gallon)")
            
            # 결과 해석
            with st.expander("📖 결과 해석") : 
                st.write(f"""
                         **입력값**: {latest['weight']} lbs
                         
                         **예측 연비**: {latest['predicted_mpg'] : .2f} mpg
                         
                         **해석**
                         - 무게가 {latest['weight']} lbs인 자동차의 예상 연비는 {latest['predicted_mpg'] :.2f} mpg입니다.
                         - 이 값은 학습 데이터의 패턴을 기반으로 계산되었습니다.
                         """)
                if model_info : 
                    # 모델 정보가 불러와져 있는 경우 출력
                    st.write(f"""
                             **오차 범위**
                             - 모델의 RMSE는 {model_info['rmse'] : .2f} mpg입니다.
                             - 실제 연비는 예측값에서 평균적으로 ±{model_info['rmse'] : .2f} mpg 정도 차이날 수 있습니다.
                             """)
            
# 탭 2: 모델 정보
with tab2:
    st.header("모델 정보")
    
    col1, col2 = st.columns(2)
        
    with col1:
        st.subheader("📊 성능 지표")
        st.metric("결정계수 (R²)", f"{model_info['r2_score'] : .3f}")
        st.metric("평균제곱오차 (MSE)", f"{model_info['mse'] : .3f}")
        st.metric("루트평균제곱오차 (RMSE)", f"{model_info['rmse'] : .3f}")
        
    with col2:
        st.subheader("📐 회귀식")
        st.latex(f"mpg = {model_info['coef'] : .4f} \\times weight + {model_info['intercept'] : .4f}")    
        # 수학식으로 회귀식 표시
        
        st.write("**계수 해석** : ")
        st.write(f"- 기울기 : {model_info['coef'] : .4f}")
        st.write(f"- 무게가 1 lbs 증가할 때마다 연비가 평균적으로 {abs(model_info['coef']) : .4f} mpg 감소")
        st.write(f"- 절편 : {model_info['intercept'] : .4f}")
        
    # 성능 지표 해석
    st.subheader("📖 성능 지표 해석")
    with st.expander("자세한 해석 보기"):
        st.write(f"""
                 **1. 결정계수 (R²) = {model_info['r2_score'] : .2f}**
                 - 모델이 연비 변동의 약 {model_info['r2_score'] * 100 : .2f}%를 설명합니다.
                 - {model_info['r2_score'] * 100 : .2f}%는 무게로 설명 가능하고, 나머지는 다른 요인에 의해 설명됩니다.
                 
                 **2. RMSE = {model_info['rmse'] : .2f}mpg**
                 - 예측값이 실제값과 평균적으로 {model_info['rmse'] : .2f} mpg 정도 차이가 납니다.
                 - 이 값이 작을수록 모델의 예측 정확도가 높습니다.
                 
                 **3. 모델의 한계**
                 - 무게 외에도 엔진 크기, 공기역학적 특성 등이 연비에 영향을 미칩니다.
                 - 더 정확한 예측을 위해서는 추가 변수가 필요할 수 있습니다.
                 """)
    

# 탭 3: 데이터 분석
with tab3:
    st.header("데이터 분석")
    
    # 샘플 데이터 생성 (실제로는 학습 데이터 사용)
    import seaborn as sns
    df = sns.load_dataset('mpg')
    df = df.dropna(subset = ['horsepower']) # 결측치 제거
    
    # 시각화
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("무게 vs 연비 산점도")
        fig_sactter = px.scatter(
            df,
            x = 'weight',
            y = 'mpg',
            title = '무게와 연비의 관계',
            labels = {'weight' : '무게 (lbs)', 'mpg' : '연비 (mpg)'},
            trendline = 'ols' # 회귀선 추가
        )
        
        # streamlit에 그래프 표시
        st.plotly_chart(fig_sactter, use_container_width=True)
    
    with col2:
        st.subheader("예측 히스토리")
        # 세션에 예측 기록이 있는 경우
        if st.session_state.get('predictions') : 
            pred_df = pd.DataFrame(st.session_state.predictions)
            st.dataframe(pred_df.tail(10), use_container_width=True)
            
        # 세션에 에측 기록이 없는 경우
        else : 
            st.info("예측을 수행하면 히스토리가 표시됩니다.")
            
    
    # 통계 정보
    st.subheader("데이터 통계")
    st.dataframe(df[['weight', 'mpg']].describe())