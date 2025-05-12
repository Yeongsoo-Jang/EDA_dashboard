# pages/basic_stats_page.py - 기초 통계 페이지 UI
import streamlit as st
import pandas as pd
from analysis.basic_stats import BasicStatistics
from visualizations.basic_viz import plot_histogram, plot_boxplot, plot_bar

def show(df):
    """기초 통계 페이지를 표시합니다."""
    st.title("📊 기초 통계 분석")
    
    # BasicStatistics 객체 생성
    stats_analyzer = BasicStatistics(df)
    
    # 수치형 변수 통계
    st.header("수치형 변수 통계")
    stats_df = stats_analyzer.get_basic_stats()
    st.dataframe(stats_df)
    
    # 범주형 변수 통계
    categorical_stats = stats_analyzer.get_categorical_stats()
    if not categorical_stats.empty:
        st.header("범주형 변수 통계")
        st.dataframe(categorical_stats)
    
    # 나머지 코드는 그대로 유지
    # 변수별 통계 시각화
    st.header("변수별 통계 시각화")
    
    # 변수 선택
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    col1, col2 = st.columns(2)
    
    # 수치형 변수 시각화
    if numeric_cols:
        with col1:
            st.subheader("수치형 변수 분포")
            selected_num_var = st.selectbox("변수 선택", numeric_cols)
            
            if selected_num_var:
                tab1, tab2 = st.tabs(["히스토그램", "박스플롯"])
                
                with tab1:
                    fig = plot_histogram(df, selected_num_var)
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    fig = plot_boxplot(df, selected_num_var)
                    st.plotly_chart(fig, use_container_width=True)
    
    # 범주형 변수 시각화
    if categorical_cols:
        with col2:
            st.subheader("범주형 변수 분포")
            selected_cat_var = st.selectbox("변수 선택", categorical_cols)
            
            if selected_cat_var:
                fig = plot_bar(df, selected_cat_var)
                st.plotly_chart(fig, use_container_width=True)