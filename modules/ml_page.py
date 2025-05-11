# pages/ml_page.py - 머신러닝 모델링 페이지 UI
import streamlit as st
import pandas as pd
import numpy as np
from analysis.machine_learning import prepare_ml_data, train_regression_model, train_classification_model
import plotly.express as px

def show(df):
    """머신러닝 모델링 페이지를 표시합니다."""
    st.title("🤖 머신러닝 모델링")
    
    # 목표 변수 선택
    target_column = st.selectbox(
        "목표 변수 선택 (예측하려는 변수)",
        df.columns.tolist()
    )
    
    if target_column:
        # 테스트 세트 비율 설정
        test_size = st.slider("테스트 세트 비율", 0.1, 0.5, 0.2, 0.05)
        
        # 모델 훈련 시작
        if st.button("모델 훈련 시작"):
            # 데이터 준비
            X_train, X_test, y_train, y_test, problem_type = prepare_ml_data(
                df, target_column, test_size=test_size
            )
            
            if problem_type == "regression":
                st.subheader("회귀 모델 (수치 예측)")
                
                # 모델 훈련 및 평가
                with st.spinner("회귀 모델 훈련 중..."):
                    results = train_regression_model(X_train, X_test, y_train, y_test)
                
                # 모델 성능 비교
                st.success("모델 훈련 완료!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("선형 회귀 RMSE", f"{results['linear_regression']['rmse']:.4f}")
                    st.metric("선형 회귀 R²", f"{results['linear_regression']['r2']:.4f}")
                with col2:
                    st.metric("랜덤 포레스트 RMSE", f"{results['random_forest']['rmse']:.4f}")
                    st.metric("랜덤 포레스트 R²", f"{results['random_forest']['r2']:.4f}")
                
                # 특성 중요도 시각화
                st.subheader("특성 중요도 (랜덤 포레스트)")
                
                fig = px.bar(
                    results['feature_importance'],
                    x='importance',
                    y='feature',
                    orientation='h',
                    title="변수 중요도",
                    template="plotly_white"
                )
                
                fig.update_layout(
                    xaxis_title="중요도",
                    yaxis_title="변수"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 실제값 vs 예측값 비교
                st.subheader("실제값 vs 예측값")
                
                # 예측값 생성
                rf_model = results['random_forest']['model']
                y_pred = rf_model.predict(X_test)
                
                comparison_df = pd.DataFrame({
                    '실제값': y_test,
                    '예측값': y_pred
                })
                
                fig = px.scatter(
                    comparison_df,
                    x='실제값',
                    y='예측값',
                    title="실제값 vs 예측값 비교",
                    template="plotly_white"
                )
                
                # 이상적인 예측선 추가
                max_val = max(comparison_df['실제값'].max(), comparison_df['예측값'].max())
                min_val = min(comparison_df['실제값'].min(), comparison_df['예측값'].min())
                
                fig.add_shape(
                    type="line",
                    x0=min_val,
                    y0=min_val,
                    x1=max_val,
                    y1=max_val,
                    line=dict(
                        color="red",
                        width=2,
                        dash="dash",
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            elif problem_type == "classification":
                st.subheader("분류 모델 (범주 예측)")
                
                # 모델 훈련 및 평가
                with st.spinner("분류 모델 훈련 중..."):
                    results = train_classification_model(X_train, X_test, y_train, y_test)
                
                # 모델 성능 비교
                st.success("모델 훈련 완료!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("로지스틱 회귀 정확도", f"{results['logistic_regression']['accuracy']:.4f}")
                with col2:
                    st.metric("랜덤 포레스트 정확도", f"{results['random_forest']['accuracy']:.4f}")
                
                # 분류 보고서
                st.subheader("분류 보고서 (랜덤 포레스트)")
                
                rf_report = results['random_forest']['classification_report']
                
                # 분류 보고서를 데이터프레임으로 변환
                report_df = pd.DataFrame(rf_report).T
                report_df = report_df.drop('support', axis=1)
                
                st.dataframe(report_df)
                
                # 특성 중요도 시각화
                st.subheader("특성 중요도 (랜덤 포레스트)")
                
                fig = px.bar(
                    results['feature_importance'],
                    x='importance',
                    y='feature',
                    orientation='h',
                    title="변수 중요도",
                    template="plotly_white"
                )
                
                fig.update_layout(
                    xaxis_title="중요도",
                    yaxis_title="변수"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.error("데이터 준비 중 오류가 발생했습니다.")
        
        else:
            st.info("목표 변수를 선택하고 '모델 훈련 시작' 버튼을 클릭하세요.")
    
    else:
        st.info("목표 변수를 선택하세요.")