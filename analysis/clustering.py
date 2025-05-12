# analysis/clustering.py - 군집화 분석 관련 클래스
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from kneed import KneeLocator
import warnings
import time

# DataAnalyzer 클래스 임포트
from analysis.basic_stats import DataAnalyzer


class ClusterAnalyzer(DataAnalyzer):
    """군집화 분석을 위한 기본 클래스"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Parameters:
        -----------
        df : DataFrame
            분석할 데이터프레임
        """
        super().__init__(df)
        self.scaler = None
        self.scaled_data = None
        self.cluster_model = None
        self.labels = None
        self.centers = None
        self.silhouette_avg = None
        self.clustered_df = None
    
    def _scale_data(self, columns: List[str] = None, scaler_type: str = 'standard') -> np.ndarray:
        """데이터 스케일링을 수행합니다.
        
        Parameters:
        -----------
        columns : list, default=None
            스케일링할 열 목록 (None이면 모든 수치형 열)
        scaler_type : str, default='standard'
            스케일러 유형 ('standard', 'robust', 'minmax')
        
        Returns:
        --------
        numpy.ndarray
            스케일링된 데이터
        """
        # 사용할 열 선택
        if columns is None:
            columns = self.numeric_cols
        else:
            # 존재하지 않는 열 제외
            columns = [col for col in columns if col in self.df.columns]
            # 수치형이 아닌 열 제외
            columns = [col for col in columns if pd.api.types.is_numeric_dtype(self.df[col])]
        
        if not columns:
            raise ValueError("스케일링할 수치형 열이 없습니다.")
        
        # 결측치 처리
        data = self.df[columns].fillna(self.df[columns].median())
        
        # 스케일러 선택
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"지원하지 않는 스케일러 유형: {scaler_type}")
        
        # 스케일링 수행
        self.scaled_data = self.scaler.fit_transform(data)
        
        return self.scaled_data
    
    def _prepare_result_df(self, labels: np.ndarray, cluster_col_name: str = '군집') -> pd.DataFrame:
        """군집화 결과와 원본 데이터를 결합한 데이터프레임을 생성합니다.
        
        Parameters:
        -----------
        labels : numpy.ndarray
            군집 레이블
        cluster_col_name : str, default='군집'
            군집 열 이름
        
        Returns:
        --------
        DataFrame
            군집화 결과 데이터프레임
        """
        if labels is None:
            raise ValueError("군집 레이블이 없습니다.")
        
        # 결과 데이터프레임 생성 (원본 데이터 복사)
        result_df = self.df.copy()
        
        # 군집 레이블 추가
        result_df[cluster_col_name] = labels
        
        return result_df
    
    def evaluate_clusters(self, labels: np.ndarray, data: np.ndarray) -> Dict[str, float]:
        """군집화 결과를 평가합니다.
        
        Parameters:
        -----------
        labels : numpy.ndarray
            군집 레이블
        data : numpy.ndarray
            평가에 사용할 데이터
        
        Returns:
        --------
        dict
            평가 지표 사전
        """
        metrics = {}
        
        # 실루엣 점수 (높을수록 좋음, -1 ~ 1)
        try:
            if len(set(labels)) > 1 and len(set(labels)) < len(labels):  # 최소 2개 이상의 군집이 필요하고, 모든 데이터가 다른 군집이면 안 됨
                metrics['silhouette_score'] = silhouette_score(data, labels)
        except:
            pass
        
        # Calinski-Harabasz 지수 (높을수록 좋음)
        try:
            if len(set(labels)) > 1:
                metrics['calinski_harabasz_score'] = calinski_harabasz_score(data, labels)
        except:
            pass
        
        # Davies-Bouldin 지수 (낮을수록 좋음)
        try:
            if len(set(labels)) > 1:
                metrics['davies_bouldin_score'] = davies_bouldin_score(data, labels)
        except:
            pass
        
        # 군집별 크기
        unique_labels = np.unique(labels)
        cluster_sizes = {f'cluster_{label}_size': np.sum(labels == label) for label in unique_labels}
        metrics.update(cluster_sizes)
        
        # 군집 개수
        metrics['n_clusters'] = len(unique_labels)
        
        # 이상치로 처리된 포인트 수 (-1 레이블은 이상치를 의미)
        if -1 in unique_labels:
            metrics['n_outliers'] = np.sum(labels == -1)
            metrics['outlier_percentage'] = np.sum(labels == -1) / len(labels) * 100
        
        return metrics
    
    def compare_clustering_algorithms(self, algorithms: List[str] = None, 
                                   n_clusters: int = 3, columns: List[str] = None,
                                   scaler_type: str = 'standard') -> pd.DataFrame:
        """여러 군집화 알고리즘을 비교합니다.
        
        Parameters:
        -----------
        algorithms : list, default=None
            비교할 알고리즘 목록 (None이면 모든 알고리즘)
        n_clusters : int, default=3
            군집 수
        columns : list, default=None
            분석할 열 목록 (None이면 모든 수치형 열)
        scaler_type : str, default='standard'
            스케일러 유형
        
        Returns:
        --------
        DataFrame
            알고리즘 비교 결과 데이터프레임
        """
        if algorithms is None:
            algorithms = ['kmeans', 'agglomerative', 'spectral', 'gaussian_mixture']
        
        # 데이터 스케일링
        scaled_data = self._scale_data(columns, scaler_type)
        
        results = []
        
        for algo in algorithms:
            # 각 알고리즘 실행
            try:
                if algo == 'kmeans':
                    clusterer = KMeans(n_clusters=n_clusters, random_state=42)
                elif algo == 'agglomerative':
                    clusterer = AgglomerativeClustering(n_clusters=n_clusters)
                elif algo == 'spectral':
                    clusterer = SpectralClustering(n_clusters=n_clusters, random_state=42)
                elif algo == 'gaussian_mixture':
                    clusterer = GaussianMixture(n_components=n_clusters, random_state=42)
                elif algo == 'dbscan':
                    # DBSCAN은 n_clusters 매개변수가 없음
                    clusterer = DBSCAN(eps=0.5, min_samples=5)
                else:
                    continue
                
                # 군집화 수행
                start_time = time.time()
                
                if algo == 'gaussian_mixture':
                    labels = clusterer.fit_predict(scaled_data)
                else:
                    labels = clusterer.fit(scaled_data).labels_
                
                end_time = time.time()
                
                # 평가
                metrics = self.evaluate_clusters(labels, scaled_data)
                
                # 결과 추가
                result = {
                    'algorithm': algo,
                    'n_clusters': metrics.get('n_clusters', n_clusters),
                    'execution_time': end_time - start_time
                }
                result.update(metrics)
                
                results.append(result)
                
            except Exception as e:
                print(f"{algo} 알고리즘 실행 중 오류 발생: {str(e)}")
        
        return pd.DataFrame(results)
    
    def get_cluster_profile(self, clustered_df: pd.DataFrame, cluster_col: str = '군집') -> Dict[str, pd.DataFrame]:
        """각 군집의 프로필(특성)을 분석합니다.
        
        Parameters:
        -----------
        clustered_df : DataFrame
            군집 레이블이 포함된 데이터프레임
        cluster_col : str, default='군집'
            군집 열 이름
        
        Returns:
        --------
        dict
            각 군집별 프로필 분석 결과
        """
        if cluster_col not in clustered_df.columns:
            raise ValueError(f"'{cluster_col}' 열이 데이터프레임에 없습니다.")
        
        # 수치형 열 추출
        numeric_cols = clustered_df.select_dtypes(include=['number']).columns.tolist()
        # 군집 열 제외
        if cluster_col in numeric_cols:
            numeric_cols.remove(cluster_col)
        
        results = {}
        
        # 각 군집별 기술 통계
        cluster_stats = clustered_df.groupby(cluster_col)[numeric_cols].agg(['mean', 'median', 'std', 'min', 'max'])
        results['cluster_stats'] = cluster_stats
        
        # 전체 평균 대비 각 군집의 특성 (Z-점수)
        cluster_means = clustered_df.groupby(cluster_col)[numeric_cols].mean()
        overall_means = clustered_df[numeric_cols].mean()
        overall_stds = clustered_df[numeric_cols].std()
        
        # Z-점수 계산
        z_scores = (cluster_means - overall_means) / overall_stds
        results['z_scores'] = z_scores
        
        # 군집 크기
        cluster_sizes = clustered_df[cluster_col].value_counts().sort_index()
        results['cluster_sizes'] = cluster_sizes
        
        # 범주형 변수의 군집별 분포
        categorical_cols = clustered_df.select_dtypes(exclude=['number']).columns.tolist()
        
        if categorical_cols:
            cat_distributions = {}
            
            for col in categorical_cols:
                if col != cluster_col:  # 군집 열 자체는 제외
                    try:
                        # 범주별 비율 계산
                        dist = clustered_df.groupby(cluster_col)[col].value_counts(normalize=True).unstack().fillna(0)
                        cat_distributions[col] = dist
                    except:
                        pass
            
            if cat_distributions:
                results['categorical_distributions'] = cat_distributions
        
        return results


class KMeansClustering(ClusterAnalyzer):
    """K-means 군집화를 위한 클래스"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Parameters:
        -----------
        df : DataFrame
            분석할 데이터프레임
        """
        super().__init__(df)
    
    @st.cache_data(ttl=3600)
    def perform_kmeans_clustering(self, n_clusters: int = 3, columns: List[str] = None, 
                                scaler_type: str = 'standard', random_state: int = 42,
                                n_init: int = 10) -> Tuple[pd.DataFrame, np.ndarray, float]:
        """K-means 군집화를 수행합니다.
        
        Parameters:
        -----------
        n_clusters : int, default=3
            군집 수
        columns : list, default=None
            분석할 열 목록 (None이면 모든 수치형 열)
        scaler_type : str, default='standard'
            스케일러 유형
        random_state : int, default=42
            난수 시드
        n_init : int, default=10
            초기 중심점 설정 시도 횟수
        
        Returns:
        --------
        tuple
            (군집화된 데이터프레임, 중심점, 실루엣 점수)
        """
        try:
            # 시작 시간 기록
            start_time = time.time()
            
            # 데이터 스케일링
            scaled_data = self._scale_data(columns, scaler_type)
            
            # K-means 군집화
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
            self.labels = kmeans.fit_predict(scaled_data)
            self.cluster_model = kmeans
            self.centers = kmeans.cluster_centers_
            
            # 실루엣 점수 계산
            if len(set(self.labels)) > 1:  # 최소 2개 이상의 군집이 있어야 함
                self.silhouette_avg = silhouette_score(scaled_data, self.labels)
            else:
                self.silhouette_avg = 0
            
            # 결과 데이터프레임 생성
            self.clustered_df = self._prepare_result_df(self.labels)
            
            # 실행 시간 기록
            execution_time = time.time() - start_time
            if 'kmeans' not in self.execution_times:
                self.execution_times['kmeans'] = []
            self.execution_times['kmeans'].append(execution_time)
            
            return self.clustered_df, self.centers, self.silhouette_avg
            
        except Exception as e:
            print(f"K-means 군집화 중 오류 발생: {str(e)}")
            return pd.DataFrame(), np.array([]), 0.0
    
    def get_optimal_clusters(self, max_clusters: int = 10, columns: List[str] = None,
                          scaler_type: str = 'standard', method: str = 'silhouette') -> Tuple[List[int], List[float], List[float]]:
        """최적의 군집 수를 찾습니다.
        
        Parameters:
        -----------
        max_clusters : int, default=10
            고려할 최대 군집 수
        columns : list, default=None
            분석할 열 목록 (None이면 모든 수치형 열)
        scaler_type : str, default='standard'
            스케일러 유형
        method : str, default='silhouette'
            최적 군집 수 결정 방법 ('silhouette', 'elbow', 'gap')
        
        Returns:
        --------
        tuple
            (군집 수 범위, 실루엣 점수, 관성)
        """
        # 데이터 스케일링
        scaled_data = self._scale_data(columns, scaler_type)
        
        # 군집 수 범위
        n_clusters_range = range(2, max_clusters + 1)
        
        # 각 평가 지표 저장을 위한 리스트
        silhouette_scores = []
        inertia_values = []
        ch_scores = []  # Calinski-Harabasz Index
        db_scores = []  # Davies-Bouldin Index
        
        for n_clusters in n_clusters_range:
            # K-means 군집화
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(scaled_data)
            
            # 관성 (Inertia) - 낮을수록 좋음
            inertia_values.append(kmeans.inertia_)
            
            # 실루엣 점수 - 높을수록 좋음
            try:
                silhouette_avg = silhouette_score(scaled_data, labels)
                silhouette_scores.append(silhouette_avg)
            except:
                silhouette_scores.append(0)
            
            # Calinski-Harabasz Index - 높을수록 좋음
            try:
                ch_score = calinski_harabasz_score(scaled_data, labels)
                ch_scores.append(ch_score)
            except:
                ch_scores.append(0)
            
            # Davies-Bouldin Index - 낮을수록 좋음
            try:
                db_score = davies_bouldin_score(scaled_data, labels)
                db_scores.append(db_score)
            except:
                db_scores.append(float('inf'))
        
        # 모든 결과를 데이터프레임으로 정리
        results_df = pd.DataFrame({
            'n_clusters': list(n_clusters_range),
            'silhouette_score': silhouette_scores,
            'inertia': inertia_values,
            'calinski_harabasz_score': ch_scores,
            'davies_bouldin_score': db_scores
        })
        
        # 엘보우 포인트 찾기 (Inertia 기준)
        try:
            kl = KneeLocator(
                list(n_clusters_range),
                inertia_values,
                curve='convex',
                direction='decreasing'
            )
            elbow_point = kl.elbow
            if elbow_point:
                results_df['elbow_point'] = [n == elbow_point for n in n_clusters_range]
        except:
            pass
        
        # 최적의 군집 수 설정
        if method == 'silhouette':
            best_n_clusters = n_clusters_range[np.argmax(silhouette_scores)]
        elif method == 'elbow':
            if 'elbow_point' in results_df.columns and any(results_df['elbow_point']):
                best_n_clusters = results_df[results_df['elbow_point']]['n_clusters'].iloc[0]
            else:
                # 엘보우 포인트를 찾지 못한 경우 휴리스틱으로 결정
                inertia_diffs = np.diff(inertia_values)
                inertia_diffs_pct = np.abs(inertia_diffs / np.array(inertia_values[:-1]))
                best_idx = np.argmax(inertia_diffs_pct < 0.1)
                best_n_clusters = n_clusters_range[best_idx]
        elif method == 'calinski_harabasz':
            best_n_clusters = n_clusters_range[np.argmax(ch_scores)]
        elif method == 'davies_bouldin':
            best_n_clusters = n_clusters_range[np.argmin(db_scores)]
        else:
            best_n_clusters = 3  # 기본값
        
        # 최적 군집 수 저장
        self.optimal_n_clusters = best_n_clusters
        
        return list(n_clusters_range), silhouette_scores, inertia_values, results_df
    
    def plot_optimal_clusters(self, max_clusters: int = 10, columns: List[str] = None,
                           scaler_type: str = 'standard') -> plt.Figure:
        """최적 군집 수 결정을 위한 그래프를 생성합니다.
        
        Parameters:
        -----------
        max_clusters : int, default=10
            고려할 최대 군집 수
        columns : list, default=None
            분석할 열 목록 (None이면 모든 수치형 열)
        scaler_type : str, default='standard'
            스케일러 유형
        
        Returns:
        --------
        matplotlib.figure.Figure
            최적 군집 수 결정 그래프
        """
        # 최적 군집 수 분석
        n_clusters_range, silhouette_scores, inertia_values, results_df = self.get_optimal_clusters(
            max_clusters=max_clusters,
            columns=columns,
            scaler_type=scaler_type
        )
        
        # 그래프 생성
        fig, axs = plt.subplots(2, 1, figsize=(10, 12))
        
        # 실루엣 점수 그래프
        axs[0].plot(n_clusters_range, silhouette_scores, 'o-', color='blue')
        axs[0].set_title('실루엣 점수 (높을수록 좋음)')
        axs[0].set_xlabel('군집 수')
        axs[0].set_ylabel('실루엣 점수')
        axs[0].grid(True)
        
        # 최적 군집 수 표시
        best_silhouette_idx = np.argmax(silhouette_scores)
        axs[0].plot(n_clusters_range[best_silhouette_idx], silhouette_scores[best_silhouette_idx], 
                   'o', markersize=10, fillstyle='none', c='red', mew=2)
        axs[0].axvline(x=n_clusters_range[best_silhouette_idx], linestyle='--', color='gray')
        axs[0].text(n_clusters_range[best_silhouette_idx], silhouette_scores[best_silhouette_idx] * 0.95,
                   f'최적 군집 수: {n_clusters_range[best_silhouette_idx]}', 
                   horizontalalignment='center')
        
        # 관성 그래프 (Elbow Method)
        axs[1].plot(n_clusters_range, inertia_values, 'o-', color='green')
        axs[1].set_title('Elbow Method (관성)')
        axs[1].set_xlabel('군집 수')
        axs[1].set_ylabel('관성')
        axs[1].grid(True)
        
        # 엘보우 포인트 표시
        if 'elbow_point' in results_df.columns and any(results_df['elbow_point']):
            elbow_idx = results_df[results_df['elbow_point']].index[0]
            axs[1].plot(n_clusters_range[elbow_idx], inertia_values[elbow_idx], 
                       'o', markersize=10, fillstyle='none', c='red', mew=2)
            axs[1].axvline(x=n_clusters_range[elbow_idx], linestyle='--', color='gray')
            axs[1].text(n_clusters_range[elbow_idx], inertia_values[elbow_idx] * 1.05,
                       f'Elbow Point: {n_clusters_range[elbow_idx]}', 
                       horizontalalignment='center')
        
        plt.tight_layout()
        return fig


class DBSCANClustering(ClusterAnalyzer):
    """DBSCAN 군집화를 위한 클래스"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Parameters:
        -----------
        df : DataFrame
            분석할 데이터프레임
        """
        super().__init__(df)
    
    @st.cache_data(ttl=3600)
    def perform_dbscan_clustering(self, eps: float = 0.5, min_samples: int = 5, 
                               columns: List[str] = None, scaler_type: str = 'standard') -> Tuple[pd.DataFrame, int]:
        """DBSCAN 군집화를 수행합니다.
        
        Parameters:
        -----------
        eps : float, default=0.5
            이웃 반경
        min_samples : int, default=5
            핵심 포인트 형성을 위한 최소 이웃 수
        columns : list, default=None
            분석할 열 목록 (None이면 모든 수치형 열)
        scaler_type : str, default='standard'
            스케일러 유형
        
        Returns:
        --------
        tuple
            (군집화된 데이터프레임, 군집 수)
        """
        try:
            # 시작 시간 기록
            start_time = time.time()
            
            # 데이터 스케일링
            scaled_data = self._scale_data(columns, scaler_type)
            
            # DBSCAN 군집화
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            self.labels = dbscan.fit_predict(scaled_data)
            self.cluster_model = dbscan
            
            # 군집 수 계산 (노이즈 포인트 제외)
            n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
            
            # 결과 데이터프레임 생성
            self.clustered_df = self._prepare_result_df(self.labels)
            
            # 중심점 계산 (DBSCAN은 중심점이 없지만, 각 군집의 평균점 계산)
            if n_clusters > 0:
                self.centers = np.array([scaled_data[self.labels == i].mean(axis=0) 
                                       for i in range(n_clusters) if i != -1])
            else:
                self.centers = np.array([])
            
            # 실루엣 점수 계산 (노이즈 포인트 제외)
            if n_clusters > 1 and np.sum(self.labels != -1) > 1:
                # 노이즈가 아닌 점들만 선택
                valid_points = scaled_data[self.labels != -1]
                valid_labels = self.labels[self.labels != -1]
                
                if len(set(valid_labels)) > 1:  # 최소 2개 이상의 군집이 있어야 함
                    self.silhouette_avg = silhouette_score(valid_points, valid_labels)
                else:
                    self.silhouette_avg = 0
            else:
                self.silhouette_avg = 0
            
            # 실행 시간 기록
            execution_time = time.time() - start_time
            if 'dbscan' not in self.execution_times:
                self.execution_times['dbscan'] = []
            self.execution_times['dbscan'].append(execution_time)
            
            return self.clustered_df, n_clusters
            
        except Exception as e:
            print(f"DBSCAN 군집화 중 오류 발생: {str(e)}")
            return pd.DataFrame(), 0
    
    def find_optimal_eps(self, min_samples: int = 5, columns: List[str] = None,
                       scaler_type: str = 'standard', n_neighbors: int = 5) -> Tuple[float, plt.Figure]:
        """DBSCAN에 적합한 eps 값을 찾습니다.
        
        Parameters:
        -----------
        min_samples : int, default=5
            핵심 포인트 형성을 위한 최소 이웃 수
        columns : list, default=None
            분석할 열 목록 (None이면 모든 수치형 열)
        scaler_type : str, default='standard'
            스케일러 유형
        n_neighbors : int, default=5
            최근접 이웃 개수
        
        Returns:
        --------
        tuple
            (최적 eps 값, K-거리 그래프)
        """
        from sklearn.neighbors import NearestNeighbors
        
        # 데이터 스케일링
        scaled_data = self._scale_data(columns, scaler_type)
        
        # 최근접 이웃 계산
        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(scaled_data)
        distances, indices = nbrs.kneighbors(scaled_data)
        
        # K번째 거리 계산 및 정렬
        k_dist = distances[:, -1]
        k_dist.sort()
        
        # 그래프 생성
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(len(k_dist)), k_dist)
        ax.set_xlabel('데이터 포인트 (정렬됨)')
        ax.set_ylabel(f'{n_neighbors}번째 최근접 이웃까지의 거리')
        ax.set_title('K-거리 그래프 (기울기가 급격히 변하는 지점이 적절한 eps)')
        ax.grid(True)
        
        # 엘보우 포인트 찾기
        try:
            kl = KneeLocator(
                range(len(k_dist)),
                k_dist,
                curve='convex',
                direction='increasing'
            )
            elbow_point = kl.elbow
            
            if elbow_point:
                optimal_eps = k_dist[elbow_point]
                
                # 그래프에 최적 eps 표시
                ax.axhline(y=optimal_eps, linestyle='--', color='red')
                ax.text(0, optimal_eps, f'최적 eps: {optimal_eps:.3f}', color='red')
                
                return optimal_eps, fig
            else:
                warnings.warn("최적 eps를 자동으로 찾을 수 없습니다.")
                # 휴리스틱: 중앙값 선택
                optimal_eps = np.median(k_dist)
                return optimal_eps, fig
        except:
            # 휴리스틱: 중앙값 선택
            optimal_eps = np.median(k_dist)
            return optimal_eps, fig


class HierarchicalClustering(ClusterAnalyzer):
    """계층적 군집화를 위한 클래스"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Parameters:
        -----------
        df : DataFrame
            분석할 데이터프레임
        """
        super().__init__(df)
    
    @st.cache_data(ttl=3600)
    def perform_hierarchical_clustering(self, n_clusters: int = 3, columns: List[str] = None,
                                      scaler_type: str = 'standard', linkage_method: str = 'ward',
                                      affinity: str = 'euclidean') -> pd.DataFrame:
        """계층적 군집화를 수행합니다.
        
        Parameters:
        -----------
        n_clusters : int, default=3
            군집 수
        columns : list, default=None
            분석할 열 목록 (None이면 모든 수치형 열)
        scaler_type : str, default='standard'
            스케일러 유형
        linkage_method : str, default='ward'
            연결 방법 ('ward', 'complete', 'average', 'single')
        affinity : str, default='euclidean'
            거리 측정 방식
        
        Returns:
        --------
        DataFrame
            군집화된 데이터프레임
        """
        try:
            # 시작 시간 기록
            start_time = time.time()
            
            # 데이터 스케일링
            scaled_data = self._scale_data(columns, scaler_type)
            
            # 계층적 군집화
            hc = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage=linkage_method,
                affinity=affinity
            )
            self.labels = hc.fit_predict(scaled_data)
            self.cluster_model = hc
            
            # 결과 데이터프레임 생성
            self.clustered_df = self._prepare_result_df(self.labels)
            
            # 각 군집의 중심점 계산
            if n_clusters > 0:
                self.centers = np.array([scaled_data[self.labels == i].mean(axis=0) 
                                       for i in range(n_clusters)])
            else:
                self.centers = np.array([])
            
            # 실루엣 점수 계산
            if n_clusters > 1:
                self.silhouette_avg = silhouette_score(scaled_data, self.labels)
            else:
                self.silhouette_avg = 0
            
            # 실행 시간 기록
            execution_time = time.time() - start_time
            if 'hierarchical' not in self.execution_times:
                self.execution_times['hierarchical'] = []
            self.execution_times['hierarchical'].append(execution_time)
            
            return self.clustered_df
            
        except Exception as e:
            print(f"계층적 군집화 중 오류 발생: {str(e)}")
            return pd.DataFrame()
    
    def plot_dendrogram(self, columns: List[str] = None, scaler_type: str = 'standard',
                      linkage_method: str = 'ward', max_display: int = 30) -> plt.Figure:
        """계층적 군집화 덴드로그램을 생성합니다.
        
        Parameters:
        -----------
        columns : list, default=None
            분석할 열 목록 (None이면 모든 수치형 열)
        scaler_type : str, default='standard'
            스케일러 유형
        linkage_method : str, default='ward'
            연결 방법 ('ward', 'complete', 'average', 'single')
        max_display : int, default=30
            표시할 최대 리프 노드 수
        
        Returns:
        --------
        matplotlib.figure.Figure
            덴드로그램
        """
        # 데이터 스케일링
        scaled_data = self._scale_data(columns, scaler_type)
        
        # 대용량 데이터 처리
        if len(scaled_data) > 1000:
            warnings.warn(f"대용량 데이터({len(scaled_data)}행)로 인해 샘플링을 수행합니다.")
            indices = np.random.choice(len(scaled_data), 1000, replace=False)
            sample_data = scaled_data[indices]
        else:
            sample_data = scaled_data
        
        # 연결 행렬 계산
        linked = linkage(sample_data, method=linkage_method)
        
        # 그래프 생성
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 덴드로그램 그리기
        dendrogram(
            linked,
            truncate_mode='lastp',  # 마지막 p개의 군집만 표시
            p=max_display,
            show_leaf_counts=True,
            ax=ax
        )
        
        ax.set_title(f'계층적 군집화 덴드로그램 ({linkage_method} 연결)')
        ax.set_xlabel('데이터 포인트')
        ax.set_ylabel('거리')
        
        plt.tight_layout()
        return fig


class GaussianMixtureClustering(ClusterAnalyzer):
    """가우시안 혼합 모델(GMM) 군집화를 위한 클래스"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Parameters:
        -----------
        df : DataFrame
            분석할 데이터프레임
        """
        super().__init__(df)
    
    @st.cache_data(ttl=3600)
    def perform_gmm_clustering(self, n_components: int = 3, columns: List[str] = None,
                            scaler_type: str = 'standard', covariance_type: str = 'full',
                            random_state: int = 42, n_init: int = 10) -> pd.DataFrame:
        """가우시안 혼합 모델(GMM) 군집화를 수행합니다.
        
        Parameters:
        -----------
        n_components : int, default=3
            컴포넌트(군집) 수
        columns : list, default=None
            분석할 열 목록 (None이면 모든 수치형 열)
        scaler_type : str, default='standard'
            스케일러 유형
        covariance_type : str, default='full'
            공분산 유형 ('full', 'tied', 'diag', 'spherical')
        random_state : int, default=42
            난수 시드
        n_init : int, default=10
            초기화 시도 횟수
        
        Returns:
        --------
        DataFrame
            군집화된 데이터프레임
        """
        try:
            # 시작 시간 기록
            start_time = time.time()
            
            # 데이터 스케일링
            scaled_data = self._scale_data(columns, scaler_type)
            
            # GMM 군집화
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type=covariance_type,
                random_state=random_state,
                n_init=n_init
            )
            self.labels = gmm.fit_predict(scaled_data)
            self.cluster_model = gmm
            self.centers = gmm.means_  # 각 컴포넌트의 평균
            
            # 결과 데이터프레임 생성
            self.clustered_df = self._prepare_result_df(self.labels)
            
            # 각 데이터 포인트가 각 군집에 속할 확률
            self.probabilities = gmm.predict_proba(scaled_data)
            
            # 실루엣 점수 계산
            if n_components > 1:
                self.silhouette_avg = silhouette_score(scaled_data, self.labels)
            else:
                self.silhouette_avg = 0
            
            # BIC, AIC 점수 저장
            self.bic = gmm.bic(scaled_data)
            self.aic = gmm.aic(scaled_data)
            
            # 실행 시간 기록
            execution_time = time.time() - start_time
            if 'gmm' not in self.execution_times:
                self.execution_times['gmm'] = []
            self.execution_times['gmm'].append(execution_time)
            
            return self.clustered_df
            
        except Exception as e:
            print(f"GMM 군집화 중 오류 발생: {str(e)}")
            return pd.DataFrame()
    
    def find_optimal_components(self, max_components: int = 10, columns: List[str] = None,
                             scaler_type: str = 'standard', covariance_type: str = 'full') -> Tuple[int, plt.Figure]:
        """최적의 GMM 컴포넌트 수를 찾습니다.
        
        Parameters:
        -----------
        max_components : int, default=10
            고려할 최대 컴포넌트 수
        columns : list, default=None
            분석할 열 목록 (None이면 모든 수치형 열)
        scaler_type : str, default='standard'
            스케일러 유형
        covariance_type : str, default='full'
            공분산 유형
        
        Returns:
        --------
        tuple
            (최적 컴포넌트 수, BIC/AIC 그래프)
        """
        # 데이터 스케일링
        scaled_data = self._scale_data(columns, scaler_type)
        
        # 컴포넌트 수 범위
        n_components_range = range(1, max_components + 1)
        
        # BIC, AIC 점수 저장
        bic_scores = []
        aic_scores = []
        silhouette_scores = []
        
        for n_components in n_components_range:
            # GMM 모델 학습
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type=covariance_type,
                random_state=42
            )
            gmm.fit(scaled_data)
            
            # BIC, AIC 계산
            bic_scores.append(gmm.bic(scaled_data))
            aic_scores.append(gmm.aic(scaled_data))
            
            # 실루엣 점수 계산 (컴포넌트가 2개 이상인 경우만)
            if n_components > 1:
                labels = gmm.predict(scaled_data)
                try:
                    silhouette_avg = silhouette_score(scaled_data, labels)
                    silhouette_scores.append(silhouette_avg)
                except:
                    silhouette_scores.append(0)
            else:
                silhouette_scores.append(0)
        
        # 최적 컴포넌트 수 찾기 (BIC 기준)
        optimal_n_components = n_components_range[np.argmin(bic_scores)]
        
        # 그래프 생성
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # BIC, AIC 그래프
        ax1.plot(n_components_range, bic_scores, 'o-', label='BIC')
        ax1.plot(n_components_range, aic_scores, 's-', label='AIC')
        ax1.set_title('BIC 및 AIC 점수 (낮을수록 좋음)')
        ax1.set_xlabel('컴포넌트 수')
        ax1.set_ylabel('점수')
        ax1.legend()
        ax1.grid(True)
        
        # 최적 컴포넌트 수 표시
        ax1.axvline(x=optimal_n_components, linestyle='--', color='red')
        ax1.text(optimal_n_components, min(bic_scores), f'최적 컴포넌트 수: {optimal_n_components}',
                horizontalalignment='center', verticalalignment='bottom', color='red')
        
        # 실루엣 점수 그래프
        if len(silhouette_scores) >= 2:  # 컴포넌트가 2개 이상인 경우만
            silhouette_scores = silhouette_scores[1:]  # 첫 번째 값(0) 제외
            ax2.plot(n_components_range[1:], silhouette_scores, 'o-')
            ax2.set_title('실루엣 점수 (높을수록 좋음)')
            ax2.set_xlabel('컴포넌트 수')
            ax2.set_ylabel('실루엣 점수')
            ax2.grid(True)
            
            # 최적 실루엣 점수 컴포넌트 수
            best_silhouette_idx = np.argmax(silhouette_scores)
            best_silhouette_n = n_components_range[best_silhouette_idx + 1]  # 인덱스 조정
            
            ax2.axvline(x=best_silhouette_n, linestyle='--', color='green')
            ax2.text(best_silhouette_n, max(silhouette_scores), f'최적 실루엣 점수: {best_silhouette_n}',
                    horizontalalignment='center', verticalalignment='bottom', color='green')
        
        plt.tight_layout()
        return optimal_n_components, fig