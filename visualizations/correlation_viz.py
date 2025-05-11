# visualizations/correlation_viz.py - 상관관계 시각화 관련 함수
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from config import COLORSCALES

def plot_correlation_heatmap(df):
    """상관관계 히트맵을 생성합니다."""
    numeric_df = df.select_dtypes(include=['number'])
    corr = numeric_df.corr()
    
    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale=COLORSCALES['correlation'],
        title="변수 간 상관관계"
    )
    
    fig.update_layout(
        height=700,
        template="plotly_white"
    )
    
    return fig, corr

def plot_scatter_matrix(df, columns=None):
    """산점도 행렬을 생성합니다."""
    if columns is None or len(columns) == 0:
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 5:
            columns = numeric_cols[:5]  # 최대 5개 변수만 선택
        else:
            columns = numeric_cols
    
    fig = px.scatter_matrix(
        df,
        dimensions=columns,
        title="산점도 행렬",
        template="plotly_white"
    )
    
    fig.update_layout(
        height=700
    )
    
    return fig

def plot_feature_correlations(df, target_col):
    """목표 변수와 다른 변수 간의 상관관계를 시각화합니다."""
    numeric_df = df.select_dtypes(include=['number'])
    
    if target_col not in numeric_df.columns:
        return None, "목표 변수가 수치형이 아닙니다."
    
    corr_with_target = numeric_df.corr()[target_col].sort_values(ascending=False)
    corr_with_target = corr_with_target.drop(target_col)  # 자기 자신과의 상관관계 제거
    
    fig = px.bar(
        x=corr_with_target.index,
        y=corr_with_target.values,
        title=f"{target_col}과(와)의 상관관계",
        labels={'x': '변수', 'y': '상관계수'},
        template="plotly_white"
    )
    
    fig.update_layout(
        xaxis_title="변수",
        yaxis_title="상관계수",
        height=500
    )
    
    return fig, corr_with_target