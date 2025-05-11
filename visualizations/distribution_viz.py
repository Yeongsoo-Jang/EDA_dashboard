# visualizations/distribution_viz.py - 분포 시각화 관련 함수
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

def plot_distribution_analysis(df, column):
    """변수의 분포를 분석하는 복합 그래프를 생성합니다."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=('히스토그램', '박스플롯'))
    
    # 히스토그램
    fig.add_trace(
        go.Histogram(x=df[column], name="히스토그램"),
        row=1, col=1
    )
    
    # 박스플롯
    fig.add_trace(
        go.Box(y=df[column], name="박스플롯"),
        row=1, col=2
    )
    
    fig.update_layout(
        title=f"{column} 분포 분석",
        height=500,
        showlegend=False,
        template="plotly_white"
    )
    
    return fig

def plot_categorical_distribution(df, column):
    """범주형 변수의 분포를 시각화합니다."""
    value_counts = df[column].value_counts().reset_index()
    value_counts.columns = [column, 'count']
    
    # 너무 많은 범주가 있으면 상위 10개만 표시
    if len(value_counts) > 10:
        value_counts = value_counts.head(10)
        title = f"{column} 범주 분포 (상위 10개)"
    else:
        title = f"{column} 범주 분포"
    
    fig = px.pie(
        value_counts,
        names=column,
        values='count',
        title=title,
        template="plotly_white"
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    
    return fig

def plot_qq_plot(df, column):
    """Q-Q 플롯을 생성하여 정규성을 확인합니다."""
    from scipy import stats
    
    # 결측치 제거
    data = df[column].dropna()
    
    # 정규분포 Q-Q 플롯 데이터 계산
    quantiles, z_scores = stats.probplot(data, dist='norm', fit=False)[:2]
    
    # 이론적 직선 계산
    slope, intercept, r_value, p_value, std_err = stats.linregress(quantiles, z_scores)
    line_x = np.array([min(quantiles), max(quantiles)])
    line_y = slope * line_x + intercept
    
    # 플롯리로 Q-Q 플롯 생성
    fig = go.Figure()
    
    # 산점도 추가
    fig.add_trace(
        go.Scatter(
            x=quantiles,
            y=z_scores,
            mode='markers',
            name='데이터 포인트',
            marker=dict(color='blue')
        )
    )
    
    # 이론적 직선 추가
    fig.add_trace(
        go.Scatter(
            x=line_x,
            y=line_y,
            mode='lines',
            name='이론적 직선',
            line=dict(color='red')
        )
    )
    
    fig.update_layout(
        title=f"{column} Q-Q 플롯 (정규성 검정)",
        xaxis_title="이론적 분위수",
        yaxis_title="샘플 분위수",
        height=500,
        template="plotly_white"
    )
    
    return fig