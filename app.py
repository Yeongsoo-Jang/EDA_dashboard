# app.py - 메인 애플리케이션 진입점 (오늘의집 맞춤)
import streamlit as st
import pandas as pd
from modules import home, basic_stats_page, variable_page, advanced_page, ml_page
from utils.data_loader import load_data
from config import BRAND_COLORS

# 페이지 설정
st.set_page_config(
    page_title="EDA 데이터 분석 대시보드",
    page_icon="📊",
    layout="wide"
)

# 브랜드 색상 적용 - 사이드바
st.markdown(f"""
<style>
    [data-testid="stSidebar"] {{
        background-color: {BRAND_COLORS['tertiary']};
    }}
    [data-testid="stSidebar"] .sidebar-content {{
        color: white;
    }}
    [data-testid="stSidebarUserContent"] {{
        padding-top: 1rem;
    }}
    .stRadio {{
        background-color: white;
        border-radius: 10px;
        padding: 10px;
        margin-top: 10px;
    }}
    .stRadio label {{
        color: {BRAND_COLORS['text']};
    }}
    [data-testid="stFileUploader"] {{
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
    }}
    footer {{
        display: none !important;
    }}
</style>
""", unsafe_allow_html=True)

# 사이드바 - 페이지 네비게이션
st.sidebar.title("📊 오늘의집 데이터 분석")

# 로고 - 사이드바 상단
logo_html = f"""
<div style="text-align: center; margin-bottom: 30px; margin-top: 10px;">
    <div style="background-color: white; color: {BRAND_COLORS['primary']}; 
               padding: 1rem; border-radius: 10px; font-size: 1.2rem; font-weight: bold;">
        오늘의집 데이터 분석
    </div>
</div>
"""
st.sidebar.markdown(logo_html, unsafe_allow_html=True)

# 사이드바 - 파일 업로드
st.sidebar.markdown("### 📁 데이터 업로드")
uploaded_file = st.sidebar.file_uploader("CSV, JSON 또는 엑셀 파일을 업로드하세요", 
                                       type=['csv', 'json', 'xlsx', 'xls'],
                                       help="오늘의집 데이터를 분석하기 위한 파일을 업로드하세요.")

# 데이터 로드
if uploaded_file is not None:
    with st.spinner("데이터를 로드 중입니다..."):
        df = load_data(uploaded_file)
        st.session_state['data'] = df
        st.session_state['filename'] = uploaded_file.name
else:
    # 샘플 데이터 옵션
    if 'sample_file' in st.session_state:
        with st.spinner("샘플 데이터를 로드 중입니다..."):
            uploaded_file = st.session_state.sample_file
            df = load_data(uploaded_file)
            st.session_state['data'] = df
            st.session_state['filename'] = uploaded_file.name
    else:
        st.session_state['data'] = None

# 페이지 선택
st.sidebar.markdown("### 📑 페이지 선택")
page = st.sidebar.radio(
    "",
    ["홈", "기초 통계", "변수 분석", "고급 EDA", "머신러닝 모델링"],
    index=0,
    format_func=lambda x: {
        "홈": "🏠 홈",
        "기초 통계": "📊 기초 통계",
        "변수 분석": "📈 변수 분석",
        "고급 EDA": "🧠 고급 EDA",
        "머신러닝 모델링": "🤖 머신러닝 모델링"
    }[x]
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
st.sidebar.markdown("""
<div style="color: white; opacity: 0.7; font-size: 0.8rem; text-align: center; margin-top: 50px;">
    © 2025 오늘의집 데이터 분석 대시보드<br>
    버전 1.0.0
</div>
""", unsafe_allow_html=True)

# 가이드 정보
with st.sidebar.expander("ℹ️ 사용 가이드"):
    st.markdown("""
    - **홈**: 데이터 개요 및 주요 KPI 확인
    - **기초 통계**: 기본 통계량 및 변수별 분포
    - **변수 분석**: 개별 변수 심층 분석 및 관계 탐색
    - **고급 EDA**: PCA, 군집화, 3D 시각화 등
    - **머신러닝 모델링**: 예측 모델 구축 및 평가
    
    문의사항: data-team@yourcompany.com
    """)