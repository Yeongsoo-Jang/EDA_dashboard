# pages/home.py - 홈페이지 UI
import streamlit as st
import pandas as pd
from utils.data_loader import generate_sample_data

def show_welcome():
    """시작 화면을 표시합니다."""
    st.title("📊 비즈니스 데이터 분석 대시보드")
    
    st.markdown("""
    ### 환영합니다!
    
    이 대시보드는 비즈니스 데이터를 다양한 각도에서 분석하여 
    실용적인 인사이트를 제공합니다.
    
    **주요 기능:**
    - 📈 기초 통계 및 데이터 요약
    - 🔄 변수 간 상관관계 분석
    - 📊 변수별 심층 분석 및 시각화
    - 🧠 고급 탐색적 데이터 분석 (EDA)
    - 🤖 머신러닝을 통한 예측 모델링
    - 💡 데이터 기반 비즈니스 인사이트
    
    **시작하려면 왼쪽 사이드바에서 CSV 또는 JSON 파일을 업로드하거나
    아래에서 샘플 데이터를 선택하세요.**
    """)
    
    # 샘플 데이터 옵션
    st.subheader("샘플 데이터로 시작하기")
    
    sample_option = st.selectbox(
        "샘플 데이터 선택",
        ["직접 업로드", "판매 데이터 샘플", "고객 데이터 샘플"]
    )
    
    if sample_option != "직접 업로드":
        if st.button("샘플 데이터로 시작"):
            if sample_option == "판매 데이터 샘플":
                st.session_state.sample_file = generate_sample_data("sales")
                st.success(f"{st.session_state.sample_file.name} 샘플 데이터가 로드되었습니다!")
                st.rerun()
                
            elif sample_option == "고객 데이터 샘플":
                st.session_state.sample_file = generate_sample_data("customer")
                st.success(f"{st.session_state.sample_file.name} 샘플 데이터가 로드되었습니다!")
                st.rerun()

def show(df, filename):
    """데이터 로드 후 홈페이지를 표시합니다."""
    st.title(f"📊 {filename} 분석")
    
    # 데이터 미리보기
    st.header("📋 데이터 미리보기")
    st.write(df.head())
    
    # 데이터 정보
    st.header("ℹ️ 데이터 기본 정보")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("행 수", df.shape[0])
        st.metric("결측치가 있는 열 수", df.isna().any().sum())
    with col2:
        st.metric("열 수", df.shape[1])
        st.metric("중복된 행 수", df.duplicated().sum())
    
    # 데이터 유형 표시
    st.subheader("데이터 유형")
    dtypes_df = pd.DataFrame(df.dtypes, columns=['데이터 유형'])
    dtypes_df['결측치 수'] = df.isnull().sum()
    dtypes_df['결측치 비율 (%)'] = (df.isnull().sum() / len(df) * 100).round(2)
    dtypes_df['고유값 수'] = df.nunique()
    st.dataframe(dtypes_df)
    
    # 인사이트 미리보기
    st.header("💡 주요 인사이트 미리보기")
    
    from utils.insights import generate_basic_insights
    insights = generate_basic_insights(df)
    
    if insights:
        for i, insight in enumerate(insights[:3]):  # 상위 3개만 표시
            st.markdown(insight)
        
        if len(insights) > 3:
            st.info(f"전체 {len(insights)}개의 인사이트 중 3개를 표시했습니다. 더 많은 인사이트를 보려면 페이지를 이동하세요.")
    else:
        st.info("데이터에서 특별한 인사이트를 발견하지 못했습니다.")