import streamlit as st
import pandas as pd
import numpy as np
import time # 시간과 관련된 모듈

st.title("캐싱을 통한 성능 최적화")

st.header("1. 데이터 로드")

@st.cache_data # 데이터 처리 속도 향상
def load_data() :
  # 데이터를 로드하는 함수
  time.sleep(2) # 2초 대기
  df = pd.DataFrame(
    {'A' : np.random.rand(1000), 
    'B' : np.random.rand(1000),
    'C' : np.random.rand(1000)}
  )
  return df  

if st.button("데이터 로드") : 
  start_time = time.time()
  df = load_data() #  데이터 로드 함수 호출
  end_time = time.time()
  st.write(f"로드 시간 : {end_time - start_time : .2f}초")
  st.dataframe(df.head())


# 복잡한 계산 캐싱
st.header("2. 복잡한 계산 캐싱")

@st.cache_data 
def expensive_calculation(n) :
  '복잡한 계산 함수'
  time.sleep(1) # 1초 대기
  return sum(range(n)) # 0부터 n-1까지의 합계

n = st.number_input("숫자 입력", min_value=1, max_value=1000000, value= 100000)

if st.button("계산 실행") : 
  start_time = time.time()
  result = expensive_calculation(n)
  end_time = time.time()

  st.write(f"결과 : {result}")
  st.write(f"계산 시간 : {end_time - start_time : .2f}초")
  st.info("같은 숫자로 다시 계산하면 즉시 결과가 나옵니다(캐싱됨)")


# 모델 로드 캐싱
st.header("3. 모델 로드 캐싱")

@st.cache_resource 
def load_model() :
  time.sleep(2)
  # model = joblib.load("model.json")
  return "모델이 로드 되었습니다"

if st.button("모델 로드") :
  model = load_model()
  st.success(model)
  st.info("모델은 한번만 로드되고 이후에는 캐시에서 가져옵니다")


