import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from io import StringIO
import statsmodels.api as sm
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings('ignore')

# 페이지 설정
st.set_page_config(
    page_title="비즈니스 데이터 분석 대시보드",
    page_icon="📊",
    layout="wide"
)

# 제목 및 소개
st.title("📊 비즈니스 데이터 분석 대시보드")
st.markdown("""
이 대시보드는 비즈니스 데이터(CSV 또는 JSON)를 분석하여 다양한 인사이트를 제공합니다.
* 상관관계 분석
* 산포도 및 분포 시각화
* 기초 통계 및 요약
* 고급 EDA (탐색적 데이터 분석)
* 변수 필터링 및 사용자 정의 시각화
""")

# 사이드바 - 파일 업로드 섹션
st.sidebar.header("📁 데이터 업로드")
uploaded_file = st.sidebar.file_uploader("CSV 또는 JSON 파일을 업로드하세요", type=['csv', 'json'])

# 데이터 로드 함수
@st.cache_data
def load_data(file):
    if file.name.endswith('.csv'):
        data = pd.read_csv(file)
    elif file.name.endswith('.json'):
        data = pd.read_json(file)
    return data

# 기초 통계 함수
def get_basic_stats(df):
    numeric_df = df.select_dtypes(include=['number'])
    
    stats_df = pd.DataFrame({
        '평균': numeric_df.mean(),
        '중앙값': numeric_df.median(),
        '최소값': numeric_df.min(),
        '최대값': numeric_df.max(),
        '표준편차': numeric_df.std(),
        '결측치 수': df.isnull().sum()
    })
    
    return stats_df

# 이상치 탐지 함수
def detect_outliers(df):
    numeric_df = df.select_dtypes(include=['number'])
    outliers = {}
    
    for column in numeric_df.columns:
        Q1 = numeric_df[column].quantile(0.25)
        Q3 = numeric_df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers[column] = {
            'count': len(numeric_df[(numeric_df[column] < lower_bound) | (numeric_df[column] > upper_bound)]),
            'percentage': len(numeric_df[(numeric_df[column] < lower_bound) | (numeric_df[column] > upper_bound)]) / len(numeric_df) * 100,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
    
    return outliers

# 상관관계 분석 및 시각화 함수
def plot_correlation(df):
    numeric_df = df.select_dtypes(include=['number'])
    corr = numeric_df.corr()
    
    # Plotly 히트맵
    fig = px.imshow(
        corr, 
        text_auto=True, 
        aspect="auto",
        color_continuous_scale='RdBu_r',
        title="변수 간 상관관계"
    )
    fig.update_layout(height=700)
    
    return fig, corr

# 분포 시각화 함수
def plot_distribution(df, column):
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
        showlegend=False
    )
    
    return fig

# 산포도 함수
def plot_scatter(df, x_col, y_col, color_col=None):
    if color_col:
        fig = px.scatter(
            df, x=x_col, y=y_col, color=color_col,
            title=f"{x_col} vs {y_col} (색상: {color_col})"
        )
    else:
        fig = px.scatter(
            df, x=x_col, y=y_col,
            title=f"{x_col} vs {y_col}"
        )
    
    fig.update_layout(height=600)
    return fig

# 고급 EDA: PCA 분석
def perform_pca(df):
    numeric_df = df.select_dtypes(include=['number'])
    
    # 결측치 처리
    numeric_df = numeric_df.fillna(numeric_df.median())
    
    # 데이터 스케일링
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)
    
    # PCA 수행
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)
    
    # PCA 결과를 데이터프레임으로 변환
    pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
    
    # 원본 데이터프레임과 병합
    categorical_cols = df.select_dtypes(exclude=['number']).columns
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            pca_df[col] = df[col].values
    
    # 분산 설명 비율
    explained_variance = pca.explained_variance_ratio_
    
    return pca_df, explained_variance

# 고급 EDA: 군집화 분석
def perform_clustering(df, n_clusters=3):
    numeric_df = df.select_dtypes(include=['number'])
    
    # 결측치 처리
    numeric_df = numeric_df.fillna(numeric_df.median())
    
    # 데이터 스케일링
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)
    
    # K-means 군집화
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)
    
    # 중심점
    centers = kmeans.cluster_centers_
    
    # 군집화 결과를 원본 데이터에 추가
    result_df = df.copy()
    result_df['군집'] = clusters
    
    return result_df, centers

# EDA 인사이트 생성 함수
def generate_insights(df):
    insights = []
    
    # 1. 결측치 비율이 높은 변수 확인
    missing_percentage = df.isnull().mean() * 100
    high_missing = missing_percentage[missing_percentage > 5].index.tolist()
    
    if high_missing:
        insights.append(f"💡 다음 변수들은 결측치 비율이 5% 이상입니다: {', '.join(high_missing)}")
    
    # 2. 높은 상관관계를 가진 변수 쌍 식별
    numeric_df = df.select_dtypes(include=['number'])
    if not numeric_df.empty:
        corr = numeric_df.corr().abs()
        # 중복 제거 및 자기 자신과의 상관관계 제거
        corr_pairs = corr.unstack().sort_values(ascending=False).drop_duplicates()
        high_corr_pairs = corr_pairs[(corr_pairs > 0.7) & (corr_pairs < 1.0)]
        
        if not high_corr_pairs.empty:
            top_5_pairs = high_corr_pairs.head(5)
            for idx, corr_value in top_5_pairs.items():
                var1, var2 = idx
                insights.append(f"💡 {var1}와 {var2} 간에 높은 상관관계가 있습니다 (상관계수: {corr_value:.2f})")
    
    # 3. 왜도가 높은 변수 (분포가 비대칭인 변수)
    for col in numeric_df.columns:
        if abs(numeric_df[col].skew()) > 1:
            if numeric_df[col].skew() > 0:
                insights.append(f"💡 {col}은 오른쪽으로 치우친 분포를 보입니다 (왜도: {numeric_df[col].skew():.2f}). 로그 변환을 고려해 보세요.")
            else:
                insights.append(f"💡 {col}은 왼쪽으로 치우친 분포를 보입니다 (왜도: {numeric_df[col].skew():.2f}).")
    
    # 4. 이상치가 많은 변수
    outliers = detect_outliers(df)
    for col, stats in outliers.items():
        if stats['percentage'] > 5:
            insights.append(f"💡 {col}에는 이상치가 많습니다 ({stats['percentage']:.2f}%). 데이터 분석 시 주의가 필요합니다.")
    
    # 5. 범주형 변수 분석
    categorical_df = df.select_dtypes(exclude=['number'])
    for col in categorical_df.columns:
        if df[col].nunique() < 10 and df[col].nunique() > 1:  # 범주가 적당히 있는 경우만
            value_counts = df[col].value_counts(normalize=True) * 100
            dominant_category = value_counts.index[0]
            dominant_percentage = value_counts.iloc[0]
            
            if dominant_percentage > 70:
                insights.append(f"💡 {col}에서 '{dominant_category}'가 전체의 {dominant_percentage:.2f}%를 차지합니다. 불균형이 심합니다.")
    
    return insights

# 메인 애플리케이션 로직
if uploaded_file is not None:
    # 데이터 로드
    df = load_data(uploaded_file)
    
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
    
    # 탭 생성
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 기초 통계", "🔄 상관관계", "📈 변수 분석", "🧠 고급 EDA", "💡 인사이트"])
    
    # 탭 1: 기초 통계
    with tab1:
        st.header("기초 통계")
        stats_df = get_basic_stats(df)
        st.dataframe(stats_df)
        
        # 범주형 데이터 요약
        categorical_cols = df.select_dtypes(exclude=['number']).columns
        if len(categorical_cols) > 0:
            st.subheader("범주형 변수 요약")
            for col in categorical_cols:
                st.write(f"**{col}** 분포:")
                value_counts = df[col].value_counts()
                
                # 플롯리로 막대 그래프 생성
                fig = px.bar(
                    x=value_counts.index, 
                    y=value_counts.values,
                    title=f"{col} 범주 분포",
                    labels={'x': col, 'y': '빈도'}
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # 탭 2: 상관관계
    with tab2:
        st.header("변수 간 상관관계")
        
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            corr_fig, corr_matrix = plot_correlation(df)
            st.plotly_chart(corr_fig, use_container_width=True)
            
            # 상위 상관관계 표시
            st.subheader("상위 상관관계")
            corr_series = corr_matrix.unstack().sort_values(ascending=False)
            # 자기 자신과의 상관관계 제거 (상관계수 1.0)
            corr_series = corr_series[corr_series < 1.0]
            
            top_correlations = pd.DataFrame(corr_series.head(10), columns=['상관계수'])
            st.dataframe(top_correlations)
            
            # 산점도 행렬
            st.subheader("산점도 행렬")
            
            # 변수 선택 (최대 5개)
            if len(numeric_cols) > 5:
                selected_vars = st.multiselect(
                    "산점도 행렬에 표시할 변수를 선택하세요 (최대 5개)",
                    options=numeric_cols,
                    default=list(numeric_cols[:3])
                )
                if len(selected_vars) > 5:
                    selected_vars = selected_vars[:5]
            else:
                selected_vars = numeric_cols
            
            if selected_vars:
                fig = px.scatter_matrix(
                    df,
                    dimensions=selected_vars,
                    title="산점도 행렬"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("수치형 데이터가 없어 상관관계 분석을 수행할 수 없습니다.")
    
    # 탭 3: 변수 분석
    with tab3:
        st.header("변수별 상세 분석")
        
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
                data_type = df[selected_column].dtype
                st.write(f"**데이터 유형:** {data_type}")
            
            # 수치형 변수인 경우
            if pd.api.types.is_numeric_dtype(df[selected_column]):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("평균", df[selected_column].mean().round(2))
                    st.metric("최소값", df[selected_column].min())
                    st.metric("중앙값", df[selected_column].median())
                with col2:
                    st.metric("표준편차", df[selected_column].std().round(2))
                    st.metric("최대값", df[selected_column].max())
                    st.metric("왜도", df[selected_column].skew().round(2))
                
                # 분포 시각화
                dist_fig = plot_distribution(df, selected_column)
                st.plotly_chart(dist_fig, use_container_width=True)
                
                # 다른 변수와의 관계 분석
                st.subheader(f"{selected_column}과(와) 다른 변수와의 관계")
                
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                numeric_cols.remove(selected_column) if selected_column in numeric_cols else None
                
                if numeric_cols:
                    # 산점도 생성
                    compare_col = st.selectbox("비교할 변수 선택", numeric_cols)
                    
                    # 색상 변수 옵션 (범주형 변수만)
                    color_options = ["없음"] + df.select_dtypes(exclude=['number']).columns.tolist()
                    color_var = st.selectbox("색상으로 구분할 변수 (선택사항)", color_options)
                    
                    if compare_col:
                        if color_var != "없음":
                            scatter_fig = plot_scatter(df, selected_column, compare_col, color_var)
                        else:
                            scatter_fig = plot_scatter(df, selected_column, compare_col)
                        
                        st.plotly_chart(scatter_fig, use_container_width=True)
                else:
                    st.info("다른 수치형 변수가 없어 산점도를 생성할 수 없습니다.")
            
            # 범주형 변수인 경우
            else:
                # 범주 분포 시각화
                value_counts = df[selected_column].value_counts()
                
                # 상위 10개 범주만 표시
                if len(value_counts) > 10:
                    st.info(f"총 {len(value_counts)}개 범주 중 상위 10개만 표시합니다.")
                    value_counts = value_counts.head(10)
                
                fig = px.bar(
                    x=value_counts.index, 
                    y=value_counts.values,
                    title=f"{selected_column} 범주 분포",
                    labels={'x': selected_column, 'y': '빈도'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 다른 변수와의 관계 분석
                st.subheader(f"{selected_column}과(와) 다른 변수와의 관계")
                
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                
                if numeric_cols:
                    # 박스플롯 생성
                    numeric_var = st.selectbox("비교할 수치형 변수 선택", numeric_cols)
                    
                    if numeric_var:
                        # 범주가 너무 많은 경우 상위 범주만 선택
                        if df[selected_column].nunique() > 8:
                            top_categories = df[selected_column].value_counts().head(8).index.tolist()
                            filtered_df = df[df[selected_column].isin(top_categories)]
                            fig = px.box(
                                filtered_df, 
                                x=selected_column, 
                                y=numeric_var,
                                title=f"{selected_column}에 따른 {numeric_var} 분포 (상위 8개 범주)",
                                color=selected_column
                            )
                        else:
                            fig = px.box(
                                df, 
                                x=selected_column, 
                                y=numeric_var,
                                title=f"{selected_column}에 따른 {numeric_var} 분포",
                                color=selected_column
                            )
                        
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("수치형 변수가 없어 박스플롯을 생성할 수 없습니다.")
    
    # 탭 4: 고급 EDA
    with tab4:
        st.header("고급 탐색적 데이터 분석 (EDA)")
        
        # EDA 옵션 선택
        eda_option = st.radio(
            "분석 유형 선택",
            ["PCA (주성분 분석)", "군집화 분석"]
        )
        
        if eda_option == "PCA (주성분 분석)":
            numeric_cols = df.select_dtypes(include=['number']).columns
            
            if len(numeric_cols) < 2:
                st.warning("PCA를 수행하려면 최소 2개 이상의 수치형 변수가 필요합니다.")
            else:
                pca_df, explained_variance = perform_pca(df)
                
                # PCA 결과 시각화
                st.subheader("PCA 결과 시각화")
                
                # 색상 변수 옵션 (범주형 변수)
                color_options = ["없음"] + df.select_dtypes(exclude=['number']).columns.tolist()
                color_var = st.selectbox("PCA 결과에 색상으로 구분할 변수", color_options)
                
                if color_var != "없음" and color_var in pca_df.columns:
                    fig = px.scatter(
                        pca_df, 
                        x='PC1', 
                        y='PC2', 
                        color=color_var,
                        title="PCA 시각화",
                        labels={'PC1': f'PC1 ({explained_variance[0]*100:.2f}%)', 'PC2': f'PC2 ({explained_variance[1]*100:.2f}%)'}
                    )
                else:
                    fig = px.scatter(
                        pca_df, 
                        x='PC1', 
                        y='PC2',
                        title="PCA 시각화",
                        labels={'PC1': f'PC1 ({explained_variance[0]*100:.2f}%)', 'PC2': f'PC2 ({explained_variance[1]*100:.2f}%)'}
                    )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 분산 설명률
                st.metric("PC1 + PC2 누적 분산 설명률", f"{(explained_variance[0] + explained_variance[1])*100:.2f}%")
        
        elif eda_option == "군집화 분석":
            numeric_cols = df.select_dtypes(include=['number']).columns
            
            if len(numeric_cols) < 2:
                st.warning("군집화 분석을 수행하려면 최소 2개 이상의 수치형 변수가 필요합니다.")
            else:
                # 군집 수 선택
                n_clusters = st.slider("군집 수 선택", min_value=2, max_value=10, value=3)
                
                # 군집화 수행
                clustered_df, centers = perform_clustering(df, n_clusters)
                
                # 군집화 결과 시각화 - 2개 변수 선택
                st.subheader("군집화 결과 시각화")
                
                col1, col2 = st.columns(2)
                with col1:
                    x_var = st.selectbox("X축 변수", numeric_cols)
                with col2:
                    y_var = st.selectbox("Y축 변수", numeric_cols, index=min(1, len(numeric_cols)-1))
                
                if x_var and y_var:
                    fig = px.scatter(
                        clustered_df, 
                        x=x_var, 
                        y=y_var, 
                        color='군집',
                        title=f"군집화 결과: {x_var} vs {y_var}",
                        color_continuous_scale=px.colors.qualitative.G10
                    )
                    
                    # 군집 중심점 추가
                    x_idx = list(numeric_cols).index(x_var)
                    y_idx = list(numeric_cols).index(y_var)
                    
                    for i in range(n_clusters):
                        fig.add_trace(
                            go.Scatter(
                                x=[centers[i, x_idx]],
                                y=[centers[i, y_idx]],
                                mode='markers',
                                marker=dict(
                                    color='black',
                                    size=15,
                                    symbol='x'
                                ),
                                name=f'군집 {i} 중심'
                            )
                        )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # 군집별 통계
                st.subheader("군집별 통계")
                
                cluster_stats = clustered_df.groupby('군집')[numeric_cols].mean()
                
                # 히트맵으로 군집별 특성 시각화
                fig = px.imshow(
                    cluster_stats,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale='RdBu_r',
                    title="군집별 평균값",
                    labels=dict(x="변수", y="군집", color="값")
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 군집별 데이터 분포
                st.subheader("군집별 데이터 분포")
                selected_var = st.selectbox("분석할 변수 선택", numeric_cols)
                
                if selected_var:
                    fig = px.box(
                        clustered_df,
                        x='군집',
                        y=selected_var,
                        color='군집',
                        title=f"군집별 {selected_var} 분포"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    # 탭 5: 인사이트
    with tab5:
        st.header("데이터 인사이트")
        
        insights = generate_insights(df)
        
        if insights:
            for insight in insights:
                st.markdown(insight)
        else:
            st.info("데이터에서 특별한 인사이트를 발견하지 못했습니다.")
        
        # 추가 분석 제안
        st.subheader("추가 분석 제안")
        
        st.markdown("""
        1. **시계열 분석**: 날짜/시간 데이터가 있는 경우, 시간에 따른 트렌드를 분석해 보세요.
        2. **이상치 처리**: 확인된 이상치를 어떻게 처리할지 전략을 수립하세요.
        3. **결측치 대체**: 결측치가 많은 변수는 적절한 대체 방법을 고려하세요.
        4. **특성 공학**: 기존 변수를 활용하여 새로운 변수를 생성해 보세요.
        5. **예측 모델링**: 타겟 변수가 있다면, 머신러닝 모델을 구축해 보세요.
        """)

# 파일이 아직 업로드되지 않은 경우
else:
    st.info("CSV 또는 JSON 파일을 업로드하여 분석을 시작하세요.")
    
    # 샘플 데이터 옵션
    st.subheader("샘플 데이터로 시작하기")
    
    sample_option = st.selectbox(
        "샘플 데이터 선택",
        ["직접 업로드", "판매 데이터 샘플", "고객 데이터 샘플", "주식 데이터 샘플"]
    )
    
    if sample_option != "직접 업로드":
        if st.button("샘플 데이터로 시작"):
            if sample_option == "판매 데이터 샘플":
                # 판매 데이터 샘플 생성
                np.random.seed(42)
                dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
                
                products = ['제품A', '제품B', '제품C', '제품D']
                regions = ['서울', '부산', '대구', '인천', '광주']
                
                sales_data = []
                
                for _ in range(1000):
                    date = np.random.choice(dates)
                    product = np.random.choice(products)
                    region = np.random.choice(regions)
                    quantity = np.random.randint(1, 50)
                    price = np.random.choice([15000, 28000, 35000, 42000])
                    discount = np.random.choice([0, 0, 0, 0.1, 0.2])
                    
                    sales_data.append({
                        '날짜': date,
                        '제품': product,
                        '지역': region,
                        '수량': quantity,
                        '가격': price,
                        '할인율': discount,
                        '매출액': round(quantity * price * (1 - discount))
                    })
                
                sample_df = pd.DataFrame(sales_data)
                
                # 데이터 저장 및 재로드 (파일 업로더 대체)
                csv_str = sample_df.to_csv(index=False)
                st.session_state.sample_file = StringIO(csv_str)
                st.session_state.sample_file.name = "판매_데이터_샘플.csv"
                
                st.experimental_rerun()
                
            elif sample_option == "고객 데이터 샘플":
                # 고객 데이터 샘플 생성
                np.random.seed(42)
                
                customer_data = []
                
                for i in range(500):
                    age = np.random.randint(18, 70)
                    gender = np.random.choice(['남성', '여성'])
                    income = np.random.normal(50000, 15000)
                    spending = income * np.random.normal(0.3, 0.1) + 5000
                    visits = np.random.poisson(8)
                    satisfaction = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.1, 0.2, 0.4, 0.25])
                    membership = np.random.choice(['일반', '실버', '골드', 'VIP'], p=[0.4, 0.3, 0.2, 0.1])
                    location = np.random.choice(['서울', '경기', '인천', '부산', '대구', '기타'], p=[0.3, 0.25, 0.15, 0.1, 0.1, 0.1])
                    
                    # 상관관계를 위한 관련 필드 생성
                    if membership == 'VIP':
                        retention_years = np.random.normal(7, 2)
                    elif membership == '골드':
                        retention_years = np.random.normal(5, 2)
                    elif membership == '실버':
                        retention_years = np.random.normal(3, 1.5)
                    else:
                        retention_years = np.random.normal(1.5, 1)
                    
                    retention_years = max(0.5, retention_years)  # 최소 0.5년
                    
                    customer_data.append({
                        '고객ID': f'CUST{i+1001}',
                        '나이': age,
                        '성별': gender,
                        '연소득': round(income),
                        '연간지출액': round(spending),
                        '방문횟수': visits,
                        '만족도': satisfaction,
                        '회원등급': membership,
                        '지역': location,
                        '유지기간': round(retention_years, 1)
                    })
                
                sample_df = pd.DataFrame(customer_data)
                
                # 데이터 저장 및 재로드 (파일 업로더 대체)
                csv_str = sample_df.to_csv(index=False)
                st.session_state.sample_file = StringIO(csv_str)
                st.session_state.sample_file.name = "고객_데이터_샘플.csv"
                
                st.experimental_rerun()
                
            elif sample_option == "주식 데이터 샘플":
                # 주식 데이터 샘플 생성
                np.random.seed(42)
                
                stock_data = []
                
                # 시작 가격
                price = 50000
                volume_base = 100000
                
                # 날짜 생성
                dates = pd.date_range('2022-01-01', '2022-12-31', freq='B')  # 영업일만
                
                for date in dates:
                    # 랜덤 가격 변동
                    change_percent = np.random.normal(0.001, 0.015)
                    price = price * (1 + change_percent)
                    
                    # 계절성 추가 (Q4에 상승 경향)
                    if date.month >= 10:
                        price = price * 1.0015
                    
                    # 가격 범위
                    open_price = price * np.random.normal(1, 0.01)
                    close_price = price
                    high_price = max(open_price, close_price) * np.random.normal(1.02, 0.005)
                    low_price = min(open_price, close_price) * np.random.normal(0.98, 0.005)
                    
                    # 거래량 - 가격 변동이 클수록 거래량도 증가
                    volume = volume_base * (1 + abs(change_percent) * 10) * np.random.normal(1, 0.3)
                    
                    # 시장 지수 (상관관계를 위해)
                    market_index = 3000 + (price - 50000) * 0.03 + np.random.normal(0, 30)
                    
                    # 외국인 보유율 - 가격과 약한 상관관계
                    foreign_ownership = 30 + (price - 50000) * 0.0003 + np.random.normal(0, 3)
                    foreign_ownership = max(10, min(60, foreign_ownership))  # 10%~60% 범위 제한
                    
                    stock_data.append({
                        '날짜': date,
                        '시가': round(open_price),
                        '고가': round(high_price),
                        '저가': round(low_price),
                        '종가': round(close_price),
                        '거래량': int(volume),
                        '시장지수': round(market_index, 1),
                        '외국인보유율': round(foreign_ownership, 1),
                        '변동률': round(change_percent * 100, 2)
                    })
                
                sample_df = pd.DataFrame(stock_data)
                
                # 데이터 저장 및 재로드 (파일 업로더 대체)
                csv_str = sample_df.to_csv(index=False)
                st.session_state.sample_file = StringIO(csv_str)
                st.session_state.sample_file.name = "주식_데이터_샘플.csv"
                
                st.experimental_rerun()
    
    # 샘플 파일이 세션에 있는 경우 로드
    if 'sample_file' in st.session_state:
        uploaded_file = st.session_state.sample_file
        st.success(f"{uploaded_file.name} 샘플 데이터가 로드되었습니다!")
        st.rerun()

# 푸터
st.markdown("---")
st.markdown("© 2025 비즈니스 데이터 분석 대시보드 | Streamlit으로 제작")
