# app.py - 메인 애플리케이션 진입점
import streamlit as st
from modules import home, basic_stats_page, variable_page, advanced_page, ml_page
from utils.data_loader import load_data

# 페이지 설정
st.set_page_config(
    page_title="비즈니스 데이터 분석 대시보드",
    page_icon="📊",
    layout="wide"
)

# 사이드바 - 페이지 네비게이션
st.sidebar.title("📊 데이터 분석 대시보드")

# 사이드바 - 파일 업로드
uploaded_file = st.sidebar.file_uploader("CSV 또는 JSON 파일을 업로드하세요", type=['csv', 'json'])

# 데이터 로드
if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.session_state['data'] = df
    st.session_state['filename'] = uploaded_file.name
else:
    # 샘플 데이터 옵션
    if 'sample_file' in st.session_state:
        uploaded_file = st.session_state.sample_file
        df = load_data(uploaded_file)
        st.session_state['data'] = df
        st.session_state['filename'] = uploaded_file.name
    else:
        st.session_state['data'] = None

# 페이지 선택
page = st.sidebar.radio(
    "페이지 선택",
    ["홈", "기초 통계", "변수 분석", "고급 EDA", "머신러닝 모델링"],
    index=0
)

# 데이터가 로드된 경우에만 페이지 표시
if 'data' in st.session_state and st.session_state['data'] is not None:
    if page == "홈":
        home.show(st.session_state['data'], st.session_state['filename'])
    elif page == "기초 통계":
        basic_stats_page.show(st.session_state['data'])
    elif page == "변수 분석":
        variable_page.show(st.session_state['data'])
    elif page == "고급 EDA":
        advanced_page.show(st.session_state['data'])
    elif page == "머신러닝 모델링":
        ml_page.show(st.session_state['data'])
else:
    home.show_welcome()

# 푸터
st.sidebar.markdown("---")
st.sidebar.markdown("© 2025 비즈니스 데이터 분석 대시보드")