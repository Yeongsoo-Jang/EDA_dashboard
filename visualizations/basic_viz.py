# visualizations/basic_viz.py - 기본 시각화 관련 함수
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_histogram(df, column):
    """히스토그램을 생성합니다."""
    fig = px.histogram(
        df, 
        x=column,
        title=f"{column} 분포",
        labels={column: column, 'count': '빈도'},
        template="plotly_white"
    )
    
    fig.update_layout(
        showlegend=False,
        xaxis_title=column,
        yaxis_title="빈도"
    )
    
    return fig

def plot_boxplot(df, column):
    """박스플롯을 생성합니다."""
    fig = px.box(
        df,
        y=column,
        title=f"{column} 박스플롯",
        template="plotly_white"
    )
    
    fig.update_layout(
        showlegend=False,
        yaxis_title=column
    )
    
    return fig

def plot_bar(df, column):
    """범주형 변수에 대한 막대 그래프를 생성합니다."""
    value_counts = df[column].value_counts().reset_index()
    value_counts.columns = [column, 'count']
    
    fig = px.bar(
        value_counts,
        x=column,
        y='count',
        title=f"{column} 범주 분포",
        template="plotly_white"
    )
    
    fig.update_layout(
        xaxis_title=column,
        yaxis_title="빈도"
    )
    
    return fig

def plot_dual_axis(df, x_col, y1_col, y2_col):
    """이중 축 그래프를 생성합니다."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # 첫 번째 Y축 트레이스
    fig.add_trace(
        go.Scatter(
            x=df[x_col],
            y=df[y1_col],
            name=y1_col,
            line=dict(color='blue')
        ),
        secondary_y=False,
    )
    
    # 두 번째 Y축 트레이스
    fig.add_trace(
        go.Scatter(
            x=df[x_col],
            y=df[y2_col],
            name=y2_col,
            line=dict(color='red')
        ),
        secondary_y=True,
    )
    
    # 레이아웃 업데이트
    fig.update_layout(
        title=f"{y1_col} vs {y2_col} (기준: {x_col})",
        template="plotly_white"
    )
    
    fig.update_xaxes(title_text=x_col)
    fig.update_yaxes(title_text=y1_col, secondary_y=False)
    fig.update_yaxes(title_text=y2_col, secondary_y=True)
    
    return fig