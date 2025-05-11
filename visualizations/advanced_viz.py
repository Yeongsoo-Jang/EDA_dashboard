# visualizations/advanced_viz.py - 고급 시각화 관련 함수
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd

def plot_3d_scatter(df, x_col, y_col, z_col, color_col=None):
    """3D 산점도를 생성합니다."""
    if color_col:
        fig = px.scatter_3d(
            df, 
            x=x_col, 
            y=y_col, 
            z=z_col,
            color=color_col,
            title=f"3D 산점도: {x_col} vs {y_col} vs {z_col}",
            template="plotly_white"
        )
    else:
        fig = px.scatter_3d(
            df, 
            x=x_col, 
            y=y_col, 
            z=z_col,
            title=f"3D 산점도: {x_col} vs {y_col} vs {z_col}",
            template="plotly_white"
        )
    
    fig.update_layout(
        scene=dict(
            xaxis_title=x_col,
            yaxis_title=y_col,
            zaxis_title=z_col
        ),
        height=700
    )
    
    return fig

def plot_radar_chart(df, category_col, value_cols):
    """레이더 차트를 생성합니다."""
    # 각 범주별 평균값 계산
    grouped_df = df.groupby(category_col)[value_cols].mean().reset_index()
    
    # 레이더 차트 데이터 준비
    fig = go.Figure()
    
    for i, category in enumerate(grouped_df[category_col]):
        values = grouped_df.loc[i, value_cols].values.tolist()
        # 첫 값을 마지막에 추가하여 폐곡선 만들기
        values.append(values[0])
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=value_cols + [value_cols[0]],  # 첫 변수를 마지막에 추가
            name=category
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
            )
        ),
        title=f"{category_col}별 {', '.join(value_cols)} 특성 비교",
        height=600,
        template="plotly_white"
    )
    
    return fig

def plot_parallel_coordinates(df, cols, color_col):
    """병렬 좌표 그래프를 생성합니다."""
    fig = px.parallel_coordinates(
        df, 
        dimensions=cols,
        color=color_col,
        title=f"병렬 좌표 그래프: {color_col}에 따른 특성 비교",
        template="plotly_white"
    )
    
    fig.update_layout(
        height=600
    )
    
    return fig