# pages/variable_page.py - 변수 분석 페이지 UI
import streamlit as st
import pandas as pd
import plotly.express as px # Plotly Express 모듈 추가
from visualizations.distribution_viz import plot_distribution_analysis, plot_categorical_distribution, plot_qq_plot
from visualizations.correlation_viz import plot_scatter_matrix
from utils.data_processor import detect_outliers

def show(df):
    """변수 분석 페이지를 표시합니다."""
    st.title("📈 변수별 상세 분석")
    
    # 변수 선택
    all_cols = df.columns.tolist()
    selected_column = st.selectbox("분석할 변수 선택", all_cols)
    
    if selected_column:
        # 기본 정보 표시
        col1, col2 = st.columns(2)
        with col1:
            st.metric("고유값 수", df[selected_column].nunique())
            st.metric("결측치 수", df[selected_column].isnull().sum())
        with col2:
            st.metric("결측치 비율 (%)", (df[selected_column].isnull().sum() / len(df) * 100).round(2))
            st.write(f"**데이터 유형:** {df[selected_column].dtype}")
        
        # 수치형 변수인 경우
        if pd.api.types.is_numeric_dtype(df[selected_column]):
            st.header(f"{selected_column} 통계 정보")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("평균", df[selected_column].mean().round(2))
                st.metric("최소값", df[selected_column].min())
                st.metric("중앙값", df[selected_column].median())
            with col2:
                st.metric("표준편차", df[selected_column].std().round(2))
                st.metric("최대값", df[selected_column].max())
                st.metric("왜도", df[selected_column].skew().round(2))
            
            # 이상치 정보
            outliers = detect_outliers(df[[selected_column]])
            if selected_column in outliers:
                st.subheader("이상치 정보")
                st.write(f"이상치 수: {outliers[selected_column]['count']} ({outliers[selected_column]['percentage']:.2f}%)")
                st.write(f"하한 경계: {outliers[selected_column]['lower_bound']:.2f}")
                st.write(f"상한 경계: {outliers[selected_column]['upper_bound']:.2f}")
            
            # 분포 시각화
            st.subheader("분포 분석")
            dist_fig = plot_distribution_analysis(df, selected_column)
            st.plotly_chart(dist_fig, use_container_width=True)
            
            # 정규성 검정
            st.subheader("정규성 검정")
            qq_fig = plot_qq_plot(df, selected_column)
            st.plotly_chart(qq_fig, use_container_width=True)
            
            # 다른 변수와의 관계 분석
            st.subheader(f"{selected_column}과(와) 다른 변수와의 관계")
            
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if len(numeric_cols) > 1:  # 최소 2개 이상의 수치형 변수 필요
                # 관련 변수 선택
                related_vars = st.multiselect(
                    "함께 분석할 변수 선택",
                    [col for col in numeric_cols if col != selected_column],
                    max_selections=3
                )
                
                if related_vars:
                    # 선택한 변수와 함께 산점도 행렬 생성
                    vars_to_plot = [selected_column] + related_vars
                    scatter_fig = plot_scatter_matrix(df, vars_to_plot)
                    st.plotly_chart(scatter_fig, use_container_width=True)
        
        # 범주형 변수인 경우
        else:
            st.header(f"{selected_column} 범주 분포")
            
            # 범주 분포 시각화
            cat_fig = plot_categorical_distribution(df, selected_column)
            st.plotly_chart(cat_fig, use_container_width=True)
            
            # 수치형 변수와의 관계 분석
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            if numeric_cols:
                st.subheader(f"{selected_column}에 따른 수치형 변수 분포")
                
                # 수치형 변수 선택
                numeric_var = st.selectbox("분석할 수치형 변수 선택", numeric_cols)
                
                if numeric_var:
                    from visualizations.basic_viz import plot_boxplot
                    
                    # 범주가 너무 많은 경우 상위 범주만 선택
                    if df[selected_column].nunique() > 8:
                        st.info(f"범주가 너무 많아 상위 8개만 표시합니다. (총 {df[selected_column].nunique()}개)")
                        top_categories = df[selected_column].value_counts().head(8).index.tolist()
                        filtered_df = df[df[selected_column].isin(top_categories)]
                        
                        fig = px.box(
                            filtered_df,
                            x=selected_column,
                            y=numeric_var,
                            title=f"{selected_column}에 따른 {numeric_var} 분포 (상위 8개 범주)",
                            template="plotly_white"
                        )
                    else:
                        fig = px.box(
                            df,
                            x=selected_column,
                            y=numeric_var,
                            title=f"{selected_column}에 따른 {numeric_var} 분포",
                            template="plotly_white"
                        )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 통계적 유의성 검정
                    st.subheader("통계적 유의성 검정")
                    
                    if df[selected_column].nunique() == 2:
                        from analysis.hypothesis import perform_ttest
                        
                        result, error = perform_ttest(df, selected_column, numeric_var)
                        
                        if error:
                            st.error(error)
                        else:
                            st.write(f"**그룹1 ({result['그룹1']}) 평균:** {result['그룹1 평균']:.2f}")
                            st.write(f"**그룹2 ({result['그룹2']}) 평균:** {result['그룹2 평균']:.2f}")
                            st.write(f"**평균 차이:** {result['평균 차이']:.2f}")
                            st.write(f"**t-통계량:** {result['t-통계량']:.4f}")
                            st.write(f"**p-값:** {result['p-값']:.4f}")
                            st.write(f"**결론:** {result['결론']}")
                    
                    elif df[selected_column].nunique() > 2:
                        from analysis.hypothesis import perform_anova
                        
                        result, error = perform_anova(df, selected_column, numeric_var)
                        
                        if error:
                            st.error(error)
                        else:
                            st.write(f"**그룹 수:** {result['그룹 수']}")
                            st.write("**그룹별 평균:**")
                            for group, mean in result['그룹별 평균'].items():
                                st.write(f"- {group}: {mean:.2f}")
                            st.write(f"**F-통계량:** {result['F-통계량']:.4f}")
                            st.write(f"**p-값:** {result['p-값']:.4f}")
                            st.write(f"**결론:** {result['결론']}")