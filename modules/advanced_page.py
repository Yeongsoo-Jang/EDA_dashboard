# pages/advanced_page.py - 고급 EDA 페이지 UI
import streamlit as st
import pandas as pd
from analysis.pca import PCA
from analysis.clustering import KMeansClustering
from visualizations.advanced_viz import plot_3d_scatter, plot_radar_chart, plot_parallel_coordinates
import plotly.express as px

def show(df):
    """고급 EDA 페이지를 표시합니다."""
    st.title("🧠 고급 탐색적 데이터 분석 (EDA)")
    
    # 수치형 변수가 충분한지 확인
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.warning("고급 EDA를 수행하려면 최소 2개 이상의 수치형 변수가 필요합니다.")
        return
    
    # 분석 유형 선택
    eda_option = st.radio(
        "분석 유형 선택",
        ["PCA (주성분 분석)", "군집화 분석", "3D 시각화", "고급 시각화"]
    )
    
    if eda_option == "PCA (주성분 분석)":
        st.header("PCA (주성분 분석)")
        
        # PCA 파라미터 설정
        n_components = st.slider("주성분 개수", min_value=2, max_value=min(5, len(numeric_cols)), value=2)
        
        # PCA 객체 생성 및 PCA 수행
        pca_analyzer = PCA(df)
        pca_df, explained_variance, loadings = pca_analyzer.perform_pca(df, n_components)
        
        if explained_variance.size > 0:
            # 결과 표시
            st.subheader("PCA 결과")
            
            # 설명된 분산 비율
            import plotly.graph_objects as go # matplotlib.pyplot import was unused here
            
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=[f'PC{i+1}' for i in range(len(explained_variance))],
                y=explained_variance,
                text=[f'{v:.2%}' for v in explained_variance],
                textposition='auto'
            ))
            fig_bar.update_layout(
                title='주성분별 설명된 분산 비율',
                xaxis_title='주성분',
                yaxis_title='설명된 분산 비율',
                template='plotly_white'
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
            # 누적 설명된 분산 비율
            cumulative_variance = explained_variance.cumsum()
            st.metric("누적 설명된 분산 비율", f"{cumulative_variance[-1]:.2%}")
            
            # PCA 시각화
            st.subheader("PCA 시각화")
            
            # 색상 변수 선택
            color_options = ["없음"] + df.select_dtypes(exclude=['number']).columns.tolist()
            color_var = st.selectbox("색상으로 구분할 변수", color_options, key="pca_color_var_selector")
            
            if not pca_df.empty and 'PC1' in pca_df.columns and 'PC2' in pca_df.columns:
                if color_var != "없음" and color_var in pca_df.columns:
                    fig_scatter = px.scatter(
                        pca_df,
                        x='PC1',
                        y='PC2',
                        color=color_var,
                        title="PCA 시각화",
                        labels={'PC1': f'PC1 ({explained_variance[0]:.2%})', 'PC2': f'PC2 ({explained_variance[1]:.2%})'},
                        template="plotly_white"
                    )
                else:
                    fig_scatter = px.scatter(
                        pca_df,
                        x='PC1',
                        y='PC2',
                        title="PCA 시각화",
                        labels={'PC1': f'PC1 ({explained_variance[0]:.2%})', 'PC2': f'PC2 ({explained_variance[1]:.2%})'},
                        template="plotly_white"
                    )
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.info("PCA 시각화에 필요한 데이터가 없습니다 (PC1 또는 PC2가 생성되지 않았습니다).")
            
            # 주성분에 대한 변수 기여도
            if not loadings.empty:
                st.subheader("변수 기여도 (로딩)")
                st.dataframe(loadings)
            else:
                st.info("변수 기여도 (로딩) 정보가 없습니다.")
            
            # 바이플롯
            if n_components == 2:
                st.subheader("바이플롯 (변수와 관측치 함께 시각화)")
                
                # PCA 객체의 get_pca_biplot_data 사용
                # Pass n_components to ensure consistency if the method uses it
                biplot_pca_df, biplot_explained_variance, biplot_scaled_loadings = pca_analyzer.get_pca_biplot_data(df, n_components=n_components)
                
                if not biplot_pca_df.empty and 'PC1' in biplot_pca_df.columns and 'PC2' in biplot_pca_df.columns and \
                   biplot_explained_variance.size >= 2 and not biplot_scaled_loadings.empty:
                    fig_biplot = px.scatter(
                        biplot_pca_df,
                        x='PC1',
                        y='PC2',
                        title="PCA 바이플롯",
                        labels={'PC1': f'PC1 ({biplot_explained_variance[0]:.2%})', 'PC2': f'PC2 ({biplot_explained_variance[1]:.2%})'},
                        template="plotly_white"
                    )
                    
                    # 변수 벡터 추가
                    for i, var_name in enumerate(biplot_scaled_loadings.index):
                        fig_biplot.add_annotation(
                            x=biplot_scaled_loadings.iloc[i, 0],
                            y=biplot_scaled_loadings.iloc[i, 1],
                            ax=0,
                            ay=0,
                            xref="x",
                            yref="y",
                            axref="x",
                            ayref="y",
                            text=var_name,
                            showarrow=True,
                            arrowhead=3,
                            arrowsize=1,
                            arrowwidth=2,
                            arrowcolor='red'
                        )
                    st.plotly_chart(fig_biplot, use_container_width=True)
                else:
                    st.info("바이플롯 생성에 필요한 데이터가 충분하지 않습니다.")
        else:
            # This case handles when explained_variance.size is 0
            # The warning from perform_pca (if data was insufficient) should have already appeared.
            st.info("PCA를 수행하기에 데이터가 충분하지 않거나 오류가 발생하여 PCA 결과를 표시할 수 없습니다.")
    
    elif eda_option == "군집화 분석":
        st.header("군집화 분석")
        
        # 최적의 군집 수 찾기
        st.subheader("최적의 군집 수 선택")
        max_clusters = st.slider("최대 군집 수", min_value=2, max_value=10, value=6)
        
        # KMeansClustering 객체 생성
        kmeans_analyzer = KMeansClustering(df)
        # Adjust unpacking to accommodate potentially more than 3 return values
        n_clusters_range, silhouette_scores, inertia_values, *_ = kmeans_analyzer.get_optimal_clusters(max_clusters)
        
        # 실루엣 점수 시각화
        fig2 = px.line(
            x=n_clusters_range,
            y=silhouette_scores,
            markers=True,
            title="군집 수에 따른 실루엣 점수",
            labels={'x': '군집 수', 'y': '실루엣 점수'},
            template="plotly_white"
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # 군집 수 선택
        n_clusters = st.slider("군집 수 선택", min_value=2, max_value=max_clusters, value=3)
        
        # K-means 군집화 수행
        clustered_df, centers, silhouette_avg = kmeans_analyzer.perform_kmeans_clustering(n_clusters)
        
        st.metric("실루엣 점수", f"{silhouette_avg:.4f}")
        
        # 군집화 결과 시각화
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
                template="plotly_white",
                color_continuous_scale=px.colors.qualitative.G10
            )
            
            # 군집 중심점 추가
            import plotly.graph_objects as go
            
            # 중심점 좌표 계산
            x_idx = numeric_cols.index(x_var)
            y_idx = numeric_cols.index(y_var)
            
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
            
            st.plotly_chart(fig, use_container_width=True, key="kmeans_scatter_with_centers")
        
        # 군집별 통계
        st.subheader("군집별 통계")
        
        cluster_stats = clustered_df.groupby('군집')[numeric_cols].mean()
        
        # 히트맵으로 군집별 특성 시각화
        fig = px.imshow(
            cluster_stats,
            text_auto='.2f',
            aspect="auto",
            color_continuous_scale='RdBu_r',
            title="군집별 평균값",
            labels=dict(x="변수", y="군집", color="값"),
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True, key="kmeans_cluster_heatmap_stats")
    
    elif eda_option == "3D 시각화":
        st.header("3D 시각화")
        
        if len(numeric_cols) < 3:
            st.warning("3D 시각화를 위해서는 최소 3개의 수치형 변수가 필요합니다.")
            return
        
        col1, col2, col3 = st.columns(3)
        with col1:
            x_var = st.selectbox("X축 변수", numeric_cols)
        with col2:
            y_var = st.selectbox("Y축 변수", numeric_cols, index=min(1, len(numeric_cols)-1))
        with col3:
            z_var = st.selectbox("Z축 변수", numeric_cols, index=min(2, len(numeric_cols)-1))
        
        # 색상 변수 선택
        color_options = ["없음"] + df.select_dtypes(exclude=['number']).columns.tolist()
        color_var = st.selectbox("색상으로 구분할 변수", color_options)
        
        if x_var and y_var and z_var:
            # 3D 산점도 생성
            if color_var != "없음":
                fig = plot_3d_scatter(df, x_var, y_var, z_var, color_var)
            else:
                fig = plot_3d_scatter(df, x_var, y_var, z_var)
            
            st.plotly_chart(fig, use_container_width=True)
    
    elif eda_option == "고급 시각화":
        st.header("고급 시각화")
        
        # 시각화 유형 선택
        viz_type = st.radio(
            "시각화 유형 선택",
            ["레이더 차트", "병렬 좌표 그래프"]
        )
        
        if viz_type == "레이더 차트":
            # 범주형 변수와 수치형 변수 선택
            categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()
            
            if not categorical_cols:
                st.warning("레이더 차트를 위해서는 범주형 변수가 필요합니다.")
                return
            
            category_col = st.selectbox("범주형 변수 선택", categorical_cols)
            
            # 너무 많은 범주가 있으면 경고
            if df[category_col].nunique() > 10:
                st.warning(f"선택한 변수에 범주가 {df[category_col].nunique()}개로 너무 많습니다. 범주가 5-10개 이하인 변수를 선택하는 것이 좋습니다.")
            
            # 수치형 변수 선택
            selected_numeric = st.multiselect(
                "비교할 수치형 변수 선택 (3-7개 권장)",
                numeric_cols,
                default=numeric_cols[:min(5, len(numeric_cols))]
            )
            
            if selected_numeric and len(selected_numeric) >= 3:
                fig = plot_radar_chart(df, category_col, selected_numeric)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("레이더 차트를 생성하려면 최소 3개 이상의 수치형 변수를 선택하세요.")
        
        elif viz_type == "병렬 좌표 그래프":
            # 변수 선택
            st.subheader("병렬 좌표 그래프")
            st.write("여러 변수에 대한 데이터 포인트의 패턴을 시각화합니다.")

            # 병렬 좌표에 표시할 수치형 변수 선택
            selected_vars = st.multiselect(
                "병렬 좌표에 표시할 변수 선택",
                numeric_cols,
                default=numeric_cols[:min(5, len(numeric_cols))]
            )
            # 색상으로 구분할 변수 선택 (수치형 또는 범주가 적은 범주형)
            # 고유값 수가 너무 많은 열 (예: ID)은 제외
            MAX_CATEGORIES_FOR_COLOR = 25 # 적절한 최대 범주 수 설정 (조정 가능)
            suitable_color_cols = ["없음"] + [
                col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) or (df[col].nunique() < MAX_CATEGORIES_FOR_COLOR and df[col].dtype in ['object', 'category', 'bool'])]

            color_var = st.selectbox("색상으로 구분할 변수", suitable_color_cols, key="parallel_coords_color_var")

            if selected_vars and len(selected_vars) >= 2: # No longer require color_var to be non-None here
                # Pass None if "없음" is selected
                color_to_pass = None if color_var == "없음" else color_var
                fig = plot_parallel_coordinates(df, selected_vars, color_to_pass)
                st.plotly_chart(fig, use_container_width=True, key="parallel_coords_plot")
            else:
                st.info("병렬 좌표 그래프를 생성하려면 최소 2개 이상의 변수를 선택하세요.")
