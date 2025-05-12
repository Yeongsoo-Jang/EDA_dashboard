# visualizations/basic_viz.py - 기본 시각화 관련 함수
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import streamlit as st
from config import BRAND_COLORS, COLORSCALES, APP_CONFIG

# 캐싱을 통한 성능 최적화
@st.cache_data(ttl=3600)
def get_default_theme() -> Dict[str, Any]:
    """기본 시각화 테마 설정을 반환합니다."""
    selected_theme = st.session_state.get("theme", "default")
    colors = BRAND_COLORS.get(selected_theme, BRAND_COLORS["default"])
    
    return {
        "template": "plotly_white",
        "color_primary": colors["primary"],
        "color_secondary": colors["secondary"],
        "color_accent": colors["accent"],
        "color_text": colors["text"],
        # Provide a hardcoded fallback if "default" is not in COLORSCALES
        "colorscale": COLORSCALES.get(selected_theme, COLORSCALES.get("default", "Viridis")),
        "colorscale_diverging": COLORSCALES.get("diverging", "RdBu_r"),
        "opacity": 0.8,
        "title_font_size": 18,
        "axis_font_size": 12,
        "legend_font_size": 10,
        "margin": dict(l=50, r=50, t=80, b=50)
    }

def prepare_data_for_plotting(df: pd.DataFrame, column: str, max_groups: int = 30, 
                             max_points: int = 10000) -> pd.DataFrame:
    """시각화를 위한 데이터 전처리를 수행합니다.
    
    Parameters:
    -----------
    df : DataFrame
        시각화할 데이터프레임
    column : str
        시각화할 열
    max_groups : int, default=30
        범주형 데이터에서 표시할 최대 그룹 수
    max_points : int, default=10000
        산점도 등에 표시할 최대 데이터 포인트 수
    
    Returns:
    --------
    DataFrame
        전처리된 데이터프레임
    """
    # 대용량 데이터인 경우 샘플링
    is_large_dataset = len(df) > max_points
    if is_large_dataset:
        df = df.sample(n=max_points, random_state=42)
    
    # 범주형 변수인 경우 상위 N개 카테고리만 사용
    if pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_categorical_dtype(df[column]):
        value_counts = df[column].value_counts()
        
        if len(value_counts) > max_groups:
            top_categories = value_counts.nlargest(max_groups - 1).index.tolist()
            # 나머지 카테고리를 '기타'로 그룹화
            df_plot = df.copy()
            df_plot.loc[~df_plot[column].isin(top_categories), column] = '기타'
            
            # 정보 메시지 반환
            info_message = f"데이터에 {len(value_counts)}개의 고유 범주가 있어 상위 {max_groups-1}개와 '기타'로 그룹화되었습니다."
            return df_plot, info_message
    
    # 대용량 데이터 샘플링 메시지
    if is_large_dataset:
        return df, f"데이터가 너무 많아 {max_points}개의 샘플만 시각화합니다."
    
    return df, None

def plot_histogram(df: pd.DataFrame, column: str, bins: int = 30, color: str = None, 
                  kde: bool = False, title: str = None, height: int = 400,
                  custom_theme: Dict = None) -> go.Figure:
    """히스토그램을 생성합니다.
    
    Parameters:
    -----------
    df : DataFrame
        시각화할 데이터프레임
    column : str
        시각화할 열
    bins : int, default=30
        히스토그램 구간 수
    color : str, default=None
        막대 색상 (None이면 테마 색상 사용)
    kde : bool, default=False
        커널 밀도 추정 곡선 표시 여부
    title : str, default=None
        그래프 제목 (None이면 자동 생성)
    height : int, default=400
        그래프 높이
    custom_theme : dict, default=None
        사용자 정의 테마 설정
    
    Returns:
    --------
    plotly.graph_objects.Figure
        생성된 히스토그램 객체
    """
    theme = custom_theme or get_default_theme()
    color = color or theme["color_primary"]
    
    # 대용량 데이터 처리
    df_plot, info_message = prepare_data_for_plotting(df, column)
    
    # 자동 구간 수 조정 (데이터 특성에 따라)
    if df_plot[column].nunique() < bins:
        bins = max(10, df_plot[column].nunique())
    
    # 타이틀 설정
    if title is None:
        title = f"{column} 분포"
        if info_message:
            title += f" ({info_message})"
    
    # 히스토그램 생성
    fig = px.histogram(
        df_plot, 
        x=column,
        nbins=bins,
        title=title,
        labels={column: column, 'count': '빈도'},
        template=theme["template"],
        color_discrete_sequence=[color],
        opacity=theme["opacity"]
    )
    
    # KDE 곡선 추가 (선택 사항)
    if kde:
        from scipy.stats import gaussian_kde
        
        # 결측치 제거
        kde_data = df_plot[column].dropna()
        
        if len(kde_data) > 1:  # KDE를 계산하기 위해 최소 2개 이상의 데이터 필요
            # KDE 계산
            kde_function = gaussian_kde(kde_data)
            
            # x축 범위 지정
            x_range = np.linspace(kde_data.min(), kde_data.max(), 1000)
            
            # KDE 곡선 추가
            kde_values = kde_function(x_range)
            scaling_factor = df_plot[column].count() * (df_plot[column].max() - df_plot[column].min()) / bins
            
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=kde_values * scaling_factor,
                    mode='lines',
                    name='밀도',
                    line=dict(color=theme["color_secondary"], width=2)
                )
            )
    
    # 레이아웃 설정
    fig.update_layout(
        height=height,
        showlegend=kde,  # KDE 곡선이 있을 때만 범례 표시
        xaxis_title=column,
        yaxis_title="빈도",
        title_font=dict(size=theme["title_font_size"]),
        font=dict(size=theme["axis_font_size"]),
        margin=theme["margin"]
    )
    
    # 모바일 환경 최적화
    fig.update_layout(
        autosize=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    
    return fig

def plot_boxplot(df: pd.DataFrame, column: str, color: str = None, 
                group_by: str = None, title: str = None, height: int = 400, 
                orientation: str = 'v', points: str = 'outliers',
                custom_theme: Dict = None) -> go.Figure:
    """박스플롯을 생성합니다.
    
    Parameters:
    -----------
    df : DataFrame
        시각화할 데이터프레임
    column : str
        시각화할 열
    color : str, default=None
        박스 색상 (None이면 테마 색상 사용)
    group_by : str, default=None
        그룹화할 열 (None이면 그룹화하지 않음)
    title : str, default=None
        그래프 제목 (None이면 자동 생성)
    height : int, default=400
        그래프 높이
    orientation : str, default='v'
        박스 방향 ('v' = 세로, 'h' = 가로)
    points : str, default='outliers'
        표시할 점 ('all', 'outliers', 'none')
    custom_theme : dict, default=None
        사용자 정의 테마 설정
    
    Returns:
    --------
    plotly.graph_objects.Figure
        생성된 박스플롯 객체
    """
    theme = custom_theme or get_default_theme()
    color = color or theme["color_primary"]
    
    # 데이터 전처리
    df_plot, info_message = prepare_data_for_plotting(df, column)
    
    # 타이틀 설정
    if title is None:
        title = f"{column} 박스플롯"
        if group_by:
            title = f"{column} by {group_by} 박스플롯"
        if info_message:
            title += f" ({info_message})"
    
    # 박스플롯 생성
    if group_by:
        if orientation == 'v':
            fig = px.box(
                df_plot,
                x=group_by,
                y=column,
                color=group_by,
                title=title,
                template=theme["template"],
                points=points
            )
        else:
            fig = px.box(
                df_plot,
                y=group_by,
                x=column,
                color=group_by,
                title=title,
                template=theme["template"],
                points=points
            )
    else:
        if orientation == 'v':
            fig = px.box(
                df_plot,
                y=column,
                title=title,
                template=theme["template"],
                points=points,
                color_discrete_sequence=[color]
            )
        else:
            fig = px.box(
                df_plot,
                x=column,
                title=title,
                template=theme["template"],
                points=points,
                color_discrete_sequence=[color]
            )
    
    # 레이아웃 설정
    fig.update_layout(
        height=height,
        showlegend=group_by is not None,
        title_font=dict(size=theme["title_font_size"]),
        font=dict(size=theme["axis_font_size"]),
        margin=theme["margin"]
    )
    
    # x축, y축 제목 설정
    if orientation == 'v':
        fig.update_layout(
            xaxis_title=group_by if group_by else "",
            yaxis_title=column
        )
    else:
        fig.update_layout(
            xaxis_title=column,
            yaxis_title=group_by if group_by else ""
        )
    
    # 모바일 환경 최적화
    fig.update_layout(
        autosize=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    
    return fig

def plot_bar(df: pd.DataFrame, column: str, sort_by: str = 'value', 
            ascending: bool = False, top_n: int = None, color: str = None,
            horizontal: bool = False, title: str = None, height: int = 400,
            custom_theme: Dict = None) -> go.Figure:
    """범주형 변수에 대한 막대 그래프를 생성합니다.
    
    Parameters:
    -----------
    df : DataFrame
        시각화할 데이터프레임
    column : str
        시각화할 열
    sort_by : str, default='value'
        정렬 기준 ('value' = 값, 'name' = 이름)
    ascending : bool, default=False
        오름차순 정렬 여부
    top_n : int, default=None
        표시할 상위 n개 항목 (None이면 모두 표시)
    color : str, default=None
        막대 색상 (None이면 테마 색상 사용)
    horizontal : bool, default=False
        가로 방향 막대 그래프 여부
    title : str, default=None
        그래프 제목 (None이면 자동 생성)
    height : int, default=400
        그래프 높이
    custom_theme : dict, default=None
        사용자 정의 테마 설정
    
    Returns:
    --------
    plotly.graph_objects.Figure
        생성된 막대 그래프 객체
    """
    theme = custom_theme or get_default_theme()
    color = color or theme["color_primary"]
    
    # 데이터 전처리
    df_plot, info_message = prepare_data_for_plotting(df, column)
    
    # 값 카운트 계산
    value_counts = df_plot[column].value_counts().reset_index()
    value_counts.columns = [column, 'count']
    
    # 정렬
    if sort_by == 'value':
        value_counts = value_counts.sort_values('count', ascending=ascending)
    else:  # 'name'
        value_counts = value_counts.sort_values(column, ascending=ascending)
    
    # 상위 n개 선택
    if top_n is not None and len(value_counts) > top_n:
        if ascending:
            value_counts = value_counts.head(top_n)
        else:
            value_counts = value_counts.head(top_n)
    
    # 타이틀 설정
    if title is None:
        title = f"{column} 범주 분포"
        if top_n is not None and len(df_plot[column].unique()) > top_n:
            title += f" (상위 {top_n}개)"
        if info_message:
            title += f" ({info_message})"
    
    # 막대 그래프 생성
    if horizontal:
        fig = px.bar(
            value_counts,
            y=column,
            x='count',
            title=title,
            template=theme["template"],
            color_discrete_sequence=[color],
            opacity=theme["opacity"]
        )
    else:
        fig = px.bar(
            value_counts,
            x=column,
            y='count',
            title=title,
            template=theme["template"],
            color_discrete_sequence=[color],
            opacity=theme["opacity"]
        )
    
    # 레이아웃 설정
    fig.update_layout(
        height=height,
        showlegend=False,
        title_font=dict(size=theme["title_font_size"]),
        font=dict(size=theme["axis_font_size"]),
        margin=theme["margin"]
    )
    
    # x축, y축 제목 설정
    if horizontal:
        fig.update_layout(
            xaxis_title="빈도",
            yaxis_title=column
        )
    else:
        fig.update_layout(
            xaxis_title=column,
            yaxis_title="빈도"
        )
    
    # 모바일 환경 최적화
    fig.update_layout(
        autosize=True
    )
    
    return fig

def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str, color_col: str = None, 
                size_col: str = None, hover_data: List[str] = None, 
                trend_line: bool = False, title: str = None, height: int = 500,
                custom_theme: Dict = None) -> go.Figure:
    """산점도를 생성합니다.
    
    Parameters:
    -----------
    df : DataFrame
        시각화할 데이터프레임
    x_col : str
        x축 열
    y_col : str
        y축 열
    color_col : str, default=None
        색상 구분에 사용할 열
    size_col : str, default=None
        크기 구분에 사용할 열
    hover_data : list, default=None
        마우스 오버 시 표시할 추가 열 목록
    trend_line : bool, default=False
        추세선 표시 여부
    title : str, default=None
        그래프 제목 (None이면 자동 생성)
    height : int, default=500
        그래프 높이
    custom_theme : dict, default=None
        사용자 정의 테마 설정
    
    Returns:
    --------
    plotly.graph_objects.Figure
        생성된 산점도 객체
    """
    theme = custom_theme or get_default_theme()
    
    # 데이터 전처리
    columns_to_check = [x_col, y_col]
    if color_col:
        columns_to_check.append(color_col)
    if size_col:
        columns_to_check.append(size_col)
    
    # 각 열에 대해 시각화 가능 데이터로 변환
    info_messages = []
    df_plot = df.copy()
    
    for col in columns_to_check:
        cleaned_df, info_message = prepare_data_for_plotting(df_plot, col, max_points=5000)
        df_plot = cleaned_df
        if info_message:
            info_messages.append(info_message)
    
    # 결측치 제거
    df_plot = df_plot.dropna(subset=[x_col, y_col])
    
    # 타이틀 설정
    if title is None:
        title = f"{x_col} vs {y_col} 산점도"
        if color_col:
            title += f" (색상: {color_col})"
        if info_messages:
            title += f" ({'; '.join(info_messages)})"
    
    # 산점도 생성
    fig = px.scatter(
        df_plot,
        x=x_col,
        y=y_col,
        color=color_col,
        size=size_col,
        hover_data=hover_data,
        title=title,
        template=theme["template"],
        height=height,
        opacity=theme["opacity"]
    )
    
    # 추세선 추가
    if trend_line:
        fig.update_layout(showlegend=True)
        
        # 색상별로 다른 추세선 (색상 열이 있는 경우)
        if color_col and pd.api.types.is_categorical_dtype(df_plot[color_col]) or pd.api.types.is_object_dtype(df_plot[color_col]):
            for category in df_plot[color_col].unique():
                category_df = df_plot[df_plot[color_col] == category]
                
                if len(category_df) > 1:  # 최소 2개 데이터가 필요
                    # 간단한 선형 회귀
                    x = category_df[x_col].astype(float)
                    y = category_df[y_col].astype(float)
                    
                    # 다항식 회귀 (2차)
                    coeffs = np.polyfit(x, y, 2)
                    polynomial = np.poly1d(coeffs)
                    
                    # 정렬된 x 값 생성
                    x_range = np.linspace(x.min(), x.max(), 100)
                    y_pred = polynomial(x_range)
                    
                    # 추세선 추가
                    fig.add_trace(
                        go.Scatter(
                            x=x_range,
                            y=y_pred,
                            mode='lines',
                            name=f'{category} 추세',
                            line=dict(dash='dash', width=1)
                        )
                    )
        else:
            # 색상 구분이 없는 경우 전체 데이터에 대한 추세선
            x = df_plot[x_col].astype(float)
            y = df_plot[y_col].astype(float)
            
            # 1차 선형 회귀
            coeffs = np.polyfit(x, y, 1)
            polynomial = np.poly1d(coeffs)
            
            # 정렬된 x 값 생성
            x_range = np.linspace(x.min(), x.max(), 100)
            y_pred = polynomial(x_range)
            
            # 추세선 추가
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=y_pred,
                    mode='lines',
                    name='선형 추세',
                    line=dict(color=theme["color_secondary"], dash='dash', width=2)
                )
            )
    
    # 레이아웃 설정
    fig.update_layout(
        title_font=dict(size=theme["title_font_size"]),
        font=dict(size=theme["axis_font_size"]),
        margin=theme["margin"],
        xaxis_title=x_col,
        yaxis_title=y_col
    )
    
    # 모바일 환경 최적화
    fig.update_layout(
        autosize=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    
    return fig

def plot_dual_axis(df: pd.DataFrame, x_col: str, y1_col: str, y2_col: str, 
                 title: str = None, height: int = 500, y1_type: str = 'line',
                 y2_type: str = 'line', custom_theme: Dict = None) -> go.Figure:
    """이중 축 그래프를 생성합니다.
    
    Parameters:
    -----------
    df : DataFrame
        시각화할 데이터프레임
    x_col : str
        x축 열
    y1_col : str
        첫 번째 y축 열
    y2_col : str
        두 번째 y축 열
    title : str, default=None
        그래프 제목 (None이면 자동 생성)
    height : int, default=500
        그래프 높이
    y1_type : str, default='line'
        첫 번째 y축 그래프 유형 ('line', 'bar', 'scatter')
    y2_type : str, default='line'
        두 번째 y축 그래프 유형 ('line', 'bar', 'scatter')
    custom_theme : dict, default=None
        사용자 정의 테마 설정
    
    Returns:
    --------
    plotly.graph_objects.Figure
        생성된 이중 축 그래프 객체
    """
    theme = custom_theme or get_default_theme()
    
    # 데이터 전처리
    df_plot = df.copy()
    
    # 각 열에 대해 시각화 가능 데이터로 변환
    df_plot, _ = prepare_data_for_plotting(df_plot, x_col, max_points=5000)
    
    # 결측치 제거
    df_plot = df_plot.dropna(subset=[x_col, y1_col, y2_col])
    
    # 타이틀 설정
    if title is None:
        title = f"{y1_col} vs {y2_col} (기준: {x_col})"
    
    # 이중 축 그래프 생성
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # 첫 번째 Y축 트레이스
    if y1_type == 'line':
        fig.add_trace(
            go.Scatter(
                x=df_plot[x_col],
                y=df_plot[y1_col],
                name=y1_col,
                mode='lines+markers',
                marker=dict(size=6),
                line=dict(color=theme["color_primary"], width=2)
            ),
            secondary_y=False,
        )
    elif y1_type == 'bar':
        fig.add_trace(
            go.Bar(
                x=df_plot[x_col],
                y=df_plot[y1_col],
                name=y1_col,
                marker=dict(color=theme["color_primary"], opacity=theme["opacity"])
            ),
            secondary_y=False,
        )
    elif y1_type == 'scatter':
        fig.add_trace(
            go.Scatter(
                x=df_plot[x_col],
                y=df_plot[y1_col],
                name=y1_col,
                mode='markers',
                marker=dict(color=theme["color_primary"], size=8, opacity=theme["opacity"])
            ),
            secondary_y=False,
        )
    
    # 두 번째 Y축 트레이스
    if y2_type == 'line':
        fig.add_trace(
            go.Scatter(
                x=df_plot[x_col],
                y=df_plot[y2_col],
                name=y2_col,
                mode='lines+markers',
                marker=dict(size=6),
                line=dict(color=theme["color_accent"], width=2)
            ),
            secondary_y=True,
        )
    elif y2_type == 'bar':
        fig.add_trace(
            go.Bar(
                x=df_plot[x_col],
                y=df_plot[y2_col],
                name=y2_col,
                marker=dict(color=theme["color_accent"], opacity=theme["opacity"])
            ),
            secondary_y=True,
        )
    elif y2_type == 'scatter':
        fig.add_trace(
            go.Scatter(
                x=df_plot[x_col],
                y=df_plot[y2_col],
                name=y2_col,
                mode='markers',
                marker=dict(color=theme["color_accent"], size=8, opacity=theme["opacity"])
            ),
            secondary_y=True,
        )
    
    # 레이아웃 업데이트
    fig.update_layout(
        title=title,
        template=theme["template"],
        height=height,
        title_font=dict(size=theme["title_font_size"]),
        font=dict(size=theme["axis_font_size"]),
        margin=theme["margin"],
        hovermode="x unified"
    )
    
    fig.update_xaxes(title_text=x_col)
    fig.update_yaxes(title_text=y1_col, secondary_y=False, color=theme["color_primary"])
    fig.update_yaxes(title_text=y2_col, secondary_y=True, color=theme["color_accent"])
    
    # 모바일 환경 최적화
    fig.update_layout(
        autosize=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    
    return fig

def plot_pie(df: pd.DataFrame, column: str, values: str = None, top_n: int = 8,
           color_sequential: bool = False, title: str = None, height: int = 400,
           custom_theme: Dict = None) -> go.Figure:
    """원형 차트를 생성합니다.
    
    Parameters:
    -----------
    df : DataFrame
        시각화할 데이터프레임
    column : str
        범주형 열
    values : str, default=None
        값 열 (None이면 빈도 카운트)
    top_n : int, default=8
        표시할 상위 n개 항목
    color_sequential : bool, default=False
        순차적 색상 사용 여부
    title : str, default=None
        그래프 제목 (None이면 자동 생성)
    height : int, default=400
        그래프 높이
    custom_theme : dict, default=None
        사용자 정의 테마 설정
    
    Returns:
    --------
    plotly.graph_objects.Figure
        생성된 원형 차트 객체
    """
    theme = custom_theme or get_default_theme()
    
    # 데이터 전처리
    df_plot, info_message = prepare_data_for_plotting(df, column)
    
    if values:
        # 값에 따른 그룹핑
        value_counts = df_plot.groupby(column)[values].sum().reset_index()
        value_counts = value_counts.sort_values(values, ascending=False)
        
        # 상위 N개 + 기타 처리
        if top_n and len(value_counts) > top_n:
            top_values = value_counts.head(top_n - 1)
            other_sum = value_counts.iloc[top_n-1:][values].sum()
            other_row = pd.DataFrame({column: ['기타'], values: [other_sum]})
            value_counts = pd.concat([top_values, other_row], ignore_index=True)
    else:
        # 단순 빈도 카운트
        value_counts = df_plot[column].value_counts().reset_index()
        value_counts.columns = [column, 'count']
        
        # 상위 N개 + 기타 처리
        if top_n and len(value_counts) > top_n:
            top_values = value_counts.head(top_n - 1)
            other_sum = value_counts.iloc[top_n-1:]['count'].sum()
            other_row = pd.DataFrame({column: ['기타'], 'count': [other_sum]})
            value_counts = pd.concat([top_values, other_row], ignore_index=True)
    
    # 타이틀 설정
    if title is None:
        title = f"{column} 분포"
        if values:
            title = f"{column}별 {values} 분포"
        if len(df_plot[column].unique()) > top_n:
            title += f" (상위 {top_n-1}개 + 기타)"
        if info_message:
            title += f" ({info_message})"
    
    # 색상 설정
    if color_sequential:
        color_sequence = px.colors.sequential.Blues[3:]  # 더 밝은 색상부터 시작
    else:
        color_sequence = theme["colorscale"]
    
    # 원형 차트 생성
    if values:
        fig = px.pie(
            value_counts,
            names=column,
            values=values,
            title=title,
            template=theme["template"],
            color_discrete_sequence=color_sequence,
            height=height
        )
    else:
        fig = px.pie(
            value_counts,
            names=column,
            values='count',
            title=title,
            template=theme["template"],
            color_discrete_sequence=color_sequence,
            height=height
        )
    
    # 레이아웃 설정
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='%{label}: %{value} (%{percent})',
        marker=dict(line=dict(color='white', width=1)),
        hole=0.3  # 도넛 차트 스타일
    )
    
    fig.update_layout(
        title_font=dict(size=theme["title_font_size"]),
        font=dict(size=theme["axis_font_size"]),
        margin=theme["margin"],
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig

def plot_heatmap(df: pd.DataFrame, columns: List[str] = None, title: str = None,
               height: int = 500, text_auto: bool = True, custom_theme: Dict = None) -> go.Figure:
    """상관관계 히트맵을 생성합니다.
    
    Parameters:
    -----------
    df : DataFrame
        시각화할 데이터프레임
    columns : list, default=None
        시각화할 열 목록 (None이면 모든 수치형 열)
    title : str, default=None
        그래프 제목 (None이면 자동 생성)
    height : int, default=500
        그래프 높이
    text_auto : bool, default=True
        상관계수 표시 여부
    custom_theme : dict, default=None
        사용자 정의 테마 설정
    
    Returns:
    --------
    plotly.graph_objects.Figure
        생성된 히트맵 객체
    """
    theme = custom_theme or get_default_theme()
    
    # 데이터 전처리
    if columns is None:
        numeric_df = df.select_dtypes(include=['number'])
    else:
        numeric_df = df[columns].select_dtypes(include=['number'])
    
    # 결측치가 많은 열 제외
    columns_to_use = []
    for col in numeric_df.columns:
        if numeric_df[col].isna().mean() < 0.5:  # 결측치가 50% 미만인 열만 사용
            columns_to_use.append(col)
    
    numeric_df = numeric_df[columns_to_use]
    
    # 상관관계 계산
    corr = numeric_df.corr(method='pearson')
    
    # 타이틀 설정
    if title is None:
        title = "변수 간 상관관계 히트맵"
    
    # 히트맵 생성
    fig = px.imshow(
        corr,
        text_auto=text_auto,
        aspect="auto",
        color_continuous_scale=theme["colorscale_diverging"],
        title=title,
        template=theme["template"],
        height=height,
        zmin=-1,
        zmax=1
    )
    
    # 레이아웃 설정
    fig.update_layout(
        title_font=dict(size=theme["title_font_size"]),
        font=dict(size=theme["axis_font_size"]),
        margin=theme["margin"]
    )
    
    # 텍스트 형식 설정
    if text_auto:
        fig.update_traces(
            text=corr.round(2),
            texttemplate='%{text:.2f}'
        )
    
    # 색상 바 설정
    fig.update_coloraxes(
        colorbar=dict(
            title="상관계수",
            titleside="right",
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticktext=['-1', '-0.5', '0', '0.5', '1']
        )
    )
    
    return fig

def plot_line(df: pd.DataFrame, x_col: str, y_col: str, group_col: str = None,
            markers: bool = True, title: str = None, height: int = 400,
            custom_theme: Dict = None) -> go.Figure:
    """선 그래프를 생성합니다.
    
    Parameters:
    -----------
    df : DataFrame
        시각화할 데이터프레임
    x_col : str
        x축 열
    y_col : str
        y축 열
    group_col : str, default=None
        그룹화할 열
    markers : bool, default=True
        마커 표시 여부
    title : str, default=None
        그래프 제목 (None이면 자동 생성)
    height : int, default=400
        그래프 높이
    custom_theme : dict, default=None
        사용자 정의 테마 설정
    
    Returns:
    --------
    plotly.graph_objects.Figure
        생성된 선 그래프 객체
    """
    theme = custom_theme or get_default_theme()
    
    # 데이터 전처리
    df_plot = df.copy()
    
    # 날짜 열 변환
    if pd.api.types.is_object_dtype(df_plot[x_col]) and is_date_column(x_col, df_plot[x_col]):
        try:
            df_plot[x_col] = pd.to_datetime(df_plot[x_col])
        except:
            pass
    
    # 타이틀 설정
    if title is None:
        title = f"{y_col} 추이"
        if group_col:
            title = f"{group_col}별 {y_col} 추이"
    
    # 마커 모드 설정
    mode = 'lines+markers' if markers else 'lines'
    
    # 선 그래프 생성
    if group_col:
        fig = px.line(
            df_plot,
            x=x_col,
            y=y_col,
            color=group_col,
            title=title,
            template=theme["template"],
            height=height,
            markers=markers
        )
    else:
        fig = px.line(
            df_plot,
            x=x_col,
            y=y_col,
            title=title,
            template=theme["template"],
            height=height,
            markers=markers,
            color_discrete_sequence=[theme["color_primary"]]
        )
    
    # 레이아웃 설정
    fig.update_layout(
        title_font=dict(size=theme["title_font_size"]),
        font=dict(size=theme["axis_font_size"]),
        margin=theme["margin"],
        xaxis_title=x_col,
        yaxis_title=y_col,
        hovermode="x unified"
    )
    
    # 축 형식 설정 (날짜인 경우)
    if pd.api.types.is_datetime64_dtype(df_plot[x_col]):
        date_range = (df_plot[x_col].max() - df_plot[x_col].min()).days
        
        if date_range > 365 * 2:  # 2년 이상 데이터
            fig.update_xaxes(tickformat="%Y", dtick="M12")
        elif date_range > 60:  # 2개월 이상 데이터
            fig.update_xaxes(tickformat="%b %Y")
        else:
            fig.update_xaxes(tickformat="%b %d")
    
    # 그리드 설정
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    # 모바일 환경 최적화
    fig.update_layout(
        autosize=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    
    return fig

def plot_calendar_heatmap(df: pd.DataFrame, date_col: str, value_col: str = None,
                        freq: str = 'D', title: str = None, height: int = 400,
                        custom_theme: Dict = None) -> go.Figure:
    """달력 히트맵을 생성합니다 (날짜별 값 시각화).
    
    Parameters:
    -----------
    df : DataFrame
        시각화할 데이터프레임
    date_col : str
        날짜 열
    value_col : str, default=None
        값 열 (None이면 단순 빈도)
    freq : str, default='D'
        집계 빈도 ('D'=일별, 'W'=주별, 'M'=월별)
    title : str, default=None
        그래프 제목 (None이면 자동 생성)
    height : int, default=400
        그래프 높이
    custom_theme : dict, default=None
        사용자 정의 테마 설정
    
    Returns:
    --------
    plotly.graph_objects.Figure
        생성된 달력 히트맵 객체
    """
    theme = custom_theme or get_default_theme()
    
    # 데이터 전처리
    df_plot = df.copy()
    
    # 날짜 열 변환
    if not pd.api.types.is_datetime64_dtype(df_plot[date_col]):
        df_plot[date_col] = pd.to_datetime(df_plot[date_col], errors='coerce')
    
    # 결측치 제거
    df_plot = df_plot.dropna(subset=[date_col])
    
    # 날짜별 집계
    if value_col:
        # 값이 지정된 경우 해당 값 기준 집계
        if freq == 'D':
            df_agg = df_plot.groupby(df_plot[date_col].dt.date)[value_col].sum().reset_index()
            df_agg[date_col] = pd.to_datetime(df_agg[date_col])
        else:
            df_agg = df_plot.groupby(pd.Grouper(key=date_col, freq=freq))[value_col].sum().reset_index()
    else:
        # 값이 지정되지 않은 경우 빈도 카운트
        if freq == 'D':
            df_agg = df_plot.groupby(df_plot[date_col].dt.date).size().reset_index()
            df_agg.columns = [date_col, 'count']
            df_agg[date_col] = pd.to_datetime(df_agg[date_col])
        else:
            df_agg = df_plot.groupby(pd.Grouper(key=date_col, freq=freq)).size().reset_index()
            df_agg.columns = [date_col, 'count']
            
        value_col = 'count'
    
    # 타이틀 설정
    if title is None:
        title = f"{date_col.capitalize()}별 {value_col} 분포"
    
    # 날짜 추출
    df_agg['year'] = df_agg[date_col].dt.year
    
    if freq == 'D':
        df_agg['month'] = df_agg[date_col].dt.month
        df_agg['day'] = df_agg[date_col].dt.day
        df_agg['weekday'] = df_agg[date_col].dt.weekday
        
        # 날짜 형식 설정
        date_format = "%Y-%m-%d"
        hover_template = "<b>%{x}</b><br>%{y}<br>%{z}<extra></extra>"
        
        # 히트맵 생성
        fig = px.imshow(
            df_agg.pivot_table(index='weekday', columns=[df_agg[date_col].dt.strftime('%Y-%m-%d')], values=value_col, aggfunc='sum'),
            color_continuous_scale=theme["colorscale"],
            aspect="auto",
            title=title,
            labels={
                'weekday': '요일',
                'x': '날짜',
                'color': value_col
            }
        )
        
        # 요일 레이블 설정
        weekdays = ['월', '화', '수', '목', '금', '토', '일']
        fig.update_yaxes(
            tickvals=list(range(7)),
            ticktext=weekdays
        )
        
    elif freq == 'W':
        # 주별 히트맵
        df_agg['week'] = df_agg[date_col].dt.isocalendar().week
        
        fig = px.imshow(
            df_agg.pivot_table(index='year', columns='week', values=value_col, aggfunc='sum'),
            color_continuous_scale=theme["colorscale"],
            aspect="auto",
            title=title,
            labels={
                'year': '연도',
                'week': '주차',
                'color': value_col
            }
        )
        
    else:  # 'M'
        # 월별 히트맵
        df_agg['month'] = df_agg[date_col].dt.month
        
        fig = px.imshow(
            df_agg.pivot_table(index='year', columns='month', values=value_col, aggfunc='sum'),
            color_continuous_scale=theme["colorscale"],
            aspect="auto",
            title=title,
            labels={
                'year': '연도',
                'month': '월',
                'color': value_col
            }
        )
        
        # 월 레이블 설정
        months = ['1월', '2월', '3월', '4월', '5월', '6월', '7월', '8월', '9월', '10월', '11월', '12월']
        fig.update_xaxes(
            tickvals=list(range(1, 13)),
            ticktext=months
        )
    
    # 레이아웃 설정
    fig.update_layout(
        height=height,
        title_font=dict(size=theme["title_font_size"]),
        font=dict(size=theme["axis_font_size"]),
        margin=theme["margin"]
    )
    
    return fig

# 접근성을 위한 색상 팔레트 함수
def get_colorblind_friendly_palette(n_colors: int = 8) -> List[str]:
    """색맹 친화적인 색상 팔레트를 반환합니다."""
    # 색맹 친화적인 팔레트 (Wong, 2011)
    palette = [
        '#000000',  # 검정
        '#E69F00',  # 주황
        '#56B4E9',  # 하늘색
        '#009E73',  # 초록
        '#F0E442',  # 노랑
        '#0072B2',  # 파랑
        '#D55E00',  # 적갈색
        '#CC79A7'   # 분홍
    ]
    
    # 요청된 색상 수에 맞게 반환
    return palette[:min(n_colors, len(palette))]

# 텍스트 데이터 시각화 (새로 추가)
def plot_word_cloud(text_data: Union[str, List[str]], title: str = None, 
                  max_words: int = 100, height: int = 400,
                  custom_theme: Dict = None) -> go.Figure:
    """워드 클라우드를 생성합니다.
    
    Parameters:
    -----------
    text_data : str or list
        시각화할 텍스트 데이터
    title : str, default=None
        그래프 제목 (None이면 자동 생성)
    max_words : int, default=100
        표시할 최대 단어 수
    height : int, default=400
        그래프 높이
    custom_theme : dict, default=None
        사용자 정의 테마 설정
    
    Returns:
    --------
    Image
        생성된 워드 클라우드 이미지
    """
    try:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
        from PIL import Image
        import io
        
        theme = custom_theme or get_default_theme()
        
        # 데이터 전처리
        if isinstance(text_data, list):
            text = ' '.join(text_data)
        else:
            text = text_data
        
        # 타이틀 설정
        if title is None:
            title = "워드 클라우드"
        
        # 워드 클라우드 생성
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=max_words,
            colormap='viridis',
            collocations=False
        ).generate(text)
        
        # 이미지로 변환
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title(title, fontsize=theme["title_font_size"])
        plt.tight_layout()
        
        # 이미지 버퍼 생성
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        
        return buf
    
    except ImportError:
        st.error("워드 클라우드 생성을 위해 wordcloud 패키지가 필요합니다.")
        return None

# 지리적 시각화 (새로 추가)
def plot_geo_map(df: pd.DataFrame, lat_col: str, lon_col: str, color_col: str = None,
               size_col: str = None, hover_data: List[str] = None, 
               title: str = None, height: int = 500,
               custom_theme: Dict = None) -> go.Figure:
    """지리적 맵을 생성합니다.
    
    Parameters:
    -----------
    df : DataFrame
        시각화할 데이터프레임
    lat_col : str
        위도 열
    lon_col : str
        경도 열
    color_col : str, default=None
        색상 구분에 사용할 열
    size_col : str, default=None
        크기 구분에 사용할 열
    hover_data : list, default=None
        마우스 오버 시 표시할 추가 열 목록
    title : str, default=None
        그래프 제목 (None이면 자동 생성)
    height : int, default=500
        그래프 높이
    custom_theme : dict, default=None
        사용자 정의 테마 설정
    
    Returns:
    --------
    plotly.graph_objects.Figure
        생성된 지리적 맵 객체
    """
    theme = custom_theme or get_default_theme()
    
    # 데이터 전처리
    df_plot = df.copy()
    
    # 결측치 제거
    df_plot = df_plot.dropna(subset=[lat_col, lon_col])
    
    # 타이틀 설정
    if title is None:
        title = "지리적 데이터 시각화"
        if color_col:
            title += f" (색상: {color_col})"
    
    # 맵 생성
    fig = px.scatter_mapbox(
        df_plot,
        lat=lat_col,
        lon=lon_col,
        color=color_col,
        size=size_col,
        hover_name=hover_data[0] if hover_data else None,
        hover_data=hover_data[1:] if hover_data and len(hover_data) > 1 else None,
        title=title,
        height=height,
        zoom=3,
        opacity=theme["opacity"]
    )
    
    # 맵박스 스타일 설정
    fig.update_layout(
        mapbox_style="carto-positron",
        title_font=dict(size=theme["title_font_size"]),
        font=dict(size=theme["axis_font_size"]),
        margin=theme["margin"]
    )
    
    return fig

# 여러 그래프 배치 함수 (새로 추가)
def create_subplot_figure(plots: List[go.Figure], 
                        layout: Tuple[int, int] = None, 
                        titles: List[str] = None,
                        shared_xaxes: bool = False,
                        shared_yaxes: bool = False,
                        height: int = 800,
                        custom_theme: Dict = None) -> go.Figure:
    """여러 그래프를 포함하는 서브플롯을 생성합니다.
    
    Parameters:
    -----------
    plots : list
        배치할 그래프 객체 목록
    layout : tuple, default=None
        서브플롯 레이아웃 (행, 열) (None이면 자동 계산)
    titles : list, default=None
        각 서브플롯 제목 목록
    shared_xaxes : bool, default=False
        x축 공유 여부
    shared_yaxes : bool, default=False
        y축 공유 여부
    height : int, default=800
        전체 그래프 높이
    custom_theme : dict, default=None
        사용자 정의 테마 설정
    
    Returns:
    --------
    plotly.graph_objects.Figure
        생성된 서브플롯 객체
    """
    theme = custom_theme or get_default_theme()
    
    n_plots = len(plots)
    
    # 레이아웃 자동 계산
    if layout is None:
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        layout = (n_rows, n_cols)
    else:
        n_rows, n_cols = layout
    
    # 서브플롯 생성
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=titles if titles else [f"Plot {i+1}" for i in range(n_plots)],
        shared_xaxes=shared_xaxes,
        shared_yaxes=shared_yaxes,
        vertical_spacing=0.1
    )
    
    # 각 그래프를 서브플롯에 추가
    for i, plot in enumerate(plots):
        row = i // n_cols + 1
        col = i % n_cols + 1
        
        for trace in plot.data:
            fig.add_trace(trace, row=row, col=col)
    
    # 레이아웃 설정
    fig.update_layout(
        height=height,
        template=theme["template"],
        font=dict(size=theme["axis_font_size"]),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig

# 반응형 레이아웃 보조 함수
def get_responsive_layout(num_cols: int = 2) -> List[st.delta_generator.DeltaGenerator]:
    """Streamlit에서 반응형 레이아웃을 위한 컬럼을 생성합니다."""
    cols = st.columns(num_cols)
    return cols

def is_date_column(column_name: str, series: pd.Series) -> bool:
    """열이 날짜를 포함하는지 감지합니다."""
    # 날짜 관련 키워드 확인
    date_keywords = ['date', 'time', 'year', 'month', 'day', '일자', '날짜', '연도', '월', '일']
    has_date_keyword = any(keyword in column_name.lower() for keyword in date_keywords)
    
    # 샘플 데이터가 날짜 형식인지 확인
    if pd.api.types.is_object_dtype(series):
        sample = series.dropna().iloc[0] if not series.dropna().empty else None
        if sample:
            try:
                pd.to_datetime(sample)
                return True
            except:
                pass
    
    return has_date_keyword