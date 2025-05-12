# analysis/pca.py - PCA 분석 관련 클래스
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA as SklearnPCA
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
import warnings
import time

# DataAnalyzer 클래스 임포트
from analysis.basic_stats import DataAnalyzer


class DimensionalityReducer(DataAnalyzer):
    """차원 축소를 위한 기본 클래스"""
    
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
        self.reduction_model = None
        self.reduced_data = None
        self.explained_variance = None
        self.components = None
        self.loadings = None
    
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
    
    def _prepare_result_df(self, categorical_cols: List[str] = None) -> pd.DataFrame:
        """차원 축소 결과와 범주형 변수를 결합한 데이터프레임을 생성합니다.
        
        Parameters:
        -----------
        categorical_cols : list, default=None
            결과 데이터프레임에 포함할 범주형 열 목록
        
        Returns:
        --------
        DataFrame
            차원 축소 결과 데이터프레임
        """
        if self.reduced_data is None:
            raise ValueError("먼저 차원 축소를 수행하세요.")
        
        # 주성분 열 이름 생성
        component_names = [f"Component{i+1}" for i in range(self.reduced_data.shape[1])]
        
        # 결과 데이터프레임 생성
        result_df = pd.DataFrame(self.reduced_data, columns=component_names)
        
        # 인덱스 유지
        result_df.index = self.df.index
        
        # 범주형 변수 포함
        if categorical_cols:
            for col in categorical_cols:
                if col in self.df.columns:
                    result_df[col] = self.df[col].values
        
        return result_df


class PCA(DimensionalityReducer):
    """주성분 분석(PCA)을 위한 클래스"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Parameters:
        -----------
        df : DataFrame
            분석할 데이터프레임
        """
        super().__init__(df)
    
    @st.cache_data(ttl=3600)
    def perform_pca(self, n_components: int = 2, columns: List[str] = None, 
                   scaler_type: str = 'standard', svd_solver: str = 'auto') -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
        """주성분 분석(PCA)을 수행합니다.
        
        Parameters:
        -----------
        n_components : int, default=2
            주성분 개수
        columns : list, default=None
            분석할 열 목록 (None이면 모든 수치형 열)
        scaler_type : str, default='standard'
            스케일러 유형 ('standard', 'robust', 'minmax')
        svd_solver : str, default='auto'
            SVD 솔버 ('auto', 'full', 'arpack', 'randomized')
        
        Returns:
        --------
        tuple
            (PCA 결과 데이터프레임, 설명된 분산 비율, 로딩 행렬)
        """
        try:
            # 시작 시간 기록
            start_time = time.time()
            
            # 분석할 열 선택
            if columns is None:
                columns = self.numeric_cols
            else:
                # 존재하지 않는 열 제외
                columns = [col for col in columns if col in self.df.columns]
                # 수치형이 아닌 열 제외
                columns = [col for col in columns if pd.api.types.is_numeric_dtype(self.df[col])]
            
            if not columns:
                raise ValueError("분석할 수치형 열이 없습니다.")
            
            # 결측치 처리
            data = self.df[columns].fillna(self.df[columns].median())
            
            # 데이터 스케일링
            scaled_data = self._scale_data(columns, scaler_type)
            
            # PCA 수행
            pca = SklearnPCA(n_components=n_components, svd_solver=svd_solver)
            self.reduced_data = pca.fit_transform(scaled_data)
            self.components = pca.components_
            self.explained_variance = pca.explained_variance_ratio_
            self.reduction_model = pca
            
            # PCA 결과를 데이터프레임으로 변환
            pca_df = pd.DataFrame(
                data=self.reduced_data,
                columns=[f'PC{i+1}' for i in range(n_components)],
                index=data.index
            )
            
            # 범주형 열 추가
            categorical_cols = self.df.select_dtypes(exclude=['number']).columns
            if not categorical_cols.empty:
                for col in categorical_cols:
                    pca_df[col] = self.df[col].values
            
            # 로딩 행렬 계산
            loadings = pd.DataFrame(
                pca.components_.T,
                columns=[f'PC{i+1}' for i in range(n_components)],
                index=columns
            )
            
            # 실행 시간 기록
            execution_time = time.time() - start_time
            if 'pca' not in self.execution_times:
                self.execution_times['pca'] = []
            self.execution_times['pca'].append(execution_time)
            
            return pca_df, pca.explained_variance_ratio_, loadings
            
        except Exception as e:
            print(f"PCA 수행 중 오류 발생: {str(e)}")
            return pd.DataFrame(), np.array([]), pd.DataFrame()
    
    def get_pca_biplot_data(self, n_components: int = 2, columns: List[str] = None,
                          scaler_type: str = 'standard', scaling_factor: float = 0.7) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
        """PCA 바이플롯을 위한 데이터를 생성합니다.
        
        Parameters:
        -----------
        n_components : int, default=2
            주성분 개수 (바이플롯은 보통 2개 사용)
        columns : list, default=None
            분석할 열 목록 (None이면 모든 수치형 열)
        scaler_type : str, default='standard'
            스케일러 유형 ('standard', 'robust', 'minmax')
        scaling_factor : float, default=0.7
            로딩 벡터 스케일링 요소
        
        Returns:
        --------
        tuple
            (PCA 결과 데이터프레임, 설명된 분산 비율, 스케일링된 로딩 행렬)
        """
        if n_components != 2:
            warnings.warn("바이플롯은 일반적으로 2개의 주성분만 사용합니다.")
        
        # PCA 수행
        pca_df, explained_variance, loadings = self.perform_pca(n_components, columns, scaler_type)
        
        # 로딩 벡터 스케일링
        # 관측치와 변수를 동일한 축척으로 시각화하기 위함
        max_obs = np.max(np.abs(pca_df.iloc[:, 0:n_components].values))
        max_loading = np.max(np.abs(loadings.values))
        
        scaling = max_obs / max_loading * scaling_factor
        scaled_loadings = loadings * scaling
        
        return pca_df, explained_variance, scaled_loadings
    
    def get_pca_feature_importance(self) -> pd.DataFrame:
        """각 원본 변수의 중요도(기여도)를 계산합니다.
        
        Returns:
        --------
        DataFrame
            각 변수의 중요도가 포함된 데이터프레임
        """
        if self.components is None or self.explained_variance is None:
            raise ValueError("먼저 perform_pca 메서드를 호출하세요.")
        
        # 열 이름 가져오기 (스케일링 시 사용한 열)
        feature_names = self.numeric_cols
        
        # 각 주성분에 대한 변수 기여도 계산
        # 각 변수의 로딩을 해당 주성분의 설명된 분산 비율로 가중합하여 중요도 계산
        n_components = len(self.explained_variance)
        importance = np.zeros(len(feature_names))
        
        for i in range(n_components):
            importance += np.abs(self.components[i, :]) * self.explained_variance[i]
        
        # 결과 정규화
        importance = importance / importance.sum()
        
        # 중요도가 높은 순으로 정렬
        result_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        })
        result_df = result_df.sort_values('importance', ascending=False)
        
        return result_df
    
    def get_optimal_n_components(self, threshold: float = 0.95, max_components: int = None) -> Tuple[int, np.ndarray, pd.DataFrame]:
        """설명된 분산 비율을 기준으로 최적의 주성분 개수를 찾습니다.
        
        Parameters:
        -----------
        threshold : float, default=0.95
            목표 누적 설명된 분산 비율
        max_components : int, default=None
            고려할 최대 주성분 개수 (None이면 변수 수의 최소값)
        
        Returns:
        --------
        tuple
            (최적 주성분 개수, 누적 설명된 분산 비율, 주성분별 설명된 분산 비율 데이터프레임)
        """
        # 분석할 수치형 열
        numeric_df = self.df[self.numeric_cols].dropna()
        
        # 데이터 스케일링
        scaled_data = self._scale_data(self.numeric_cols)
        
        # 최대 주성분 수 설정
        if max_components is None:
            max_components = min(len(self.numeric_cols), len(numeric_df))
        else:
            max_components = min(max_components, len(self.numeric_cols), len(numeric_df))
        
        # PCA 수행
        pca = SklearnPCA(n_components=max_components)
        pca.fit(scaled_data)
        
        # 누적 설명된 분산 비율
        cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
        
        # 결과 데이터프레임 생성
        result_df = pd.DataFrame({
            'n_components': range(1, max_components + 1),
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance_ratio': cumulative_variance_ratio
        })
        
        # 목표 임계값을 넘는 첫 번째 주성분 개수 찾기
        optimal_n = next((i + 1 for i, ratio in enumerate(cumulative_variance_ratio) if ratio >= threshold), max_components)
        
        return optimal_n, cumulative_variance_ratio, result_df
    
    def get_component_interpretation(self, n_top_features: int = 5) -> pd.DataFrame:
        """각 주성분의 해석을 위한 주요 변수를 식별합니다.
        
        Parameters:
        -----------
        n_top_features : int, default=5
            각 주성분별로 보여줄 최상위 변수 수
        
        Returns:
        --------
        DataFrame
            각 주성분별 주요 변수 및 로딩 값
        """
        if self.components is None:
            raise ValueError("먼저 perform_pca 메서드를 호출하세요.")
        
        feature_names = self.numeric_cols
        n_components = self.components.shape[0]
        
        results = []
        
        for i in range(n_components):
            # i번째 주성분의 로딩
            loadings = self.components[i, :]
            
            # 로딩 절대값 기준 정렬
            sorted_indices = np.argsort(np.abs(loadings))[::-1]
            
            # 상위 n개 변수 선택
            top_indices = sorted_indices[:n_top_features]
            
            # 결과 추가
            for idx in top_indices:
                results.append({
                    'component': f'PC{i+1}',
                    'explained_variance': self.explained_variance[i],
                    'feature': feature_names[idx],
                    'loading': loadings[idx],
                    'loading_abs': abs(loadings[idx]),
                    'direction': 'positive' if loadings[idx] > 0 else 'negative'
                })
        
        # 결과 데이터프레임 변환
        result_df = pd.DataFrame(results)
        result_df = result_df.sort_values(['component', 'loading_abs'], ascending=[True, False])
        
        return result_df
    
    def plot_variance_explained(self, max_components: int = None) -> plt.Figure:
        """설명된 분산 비율 그래프를 생성합니다.
        
        Parameters:
        -----------
        max_components : int, default=None
            그래프에 표시할 최대 주성분 개수
        
        Returns:
        --------
        matplotlib.figure.Figure
            설명된 분산 비율 그래프
        """
        # 최적 주성분 개수 탐색
        _, cum_variance, variance_df = self.get_optimal_n_components(max_components=max_components)
        
        # 그래프 생성
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # 설명된 분산 비율 (막대 그래프)
        ax1.bar(variance_df['n_components'], variance_df['explained_variance_ratio'], 
                alpha=0.7, color='skyblue', label='개별 설명된 분산 비율')
        ax1.set_xlabel('주성분 개수')
        ax1.set_ylabel('설명된 분산 비율')
        ax1.tick_params(axis='y')
        
        # 누적 설명된 분산 비율 (선 그래프)
        ax2 = ax1.twinx()
        ax2.plot(variance_df['n_components'], variance_df['cumulative_variance_ratio'], 
                 marker='o', color='red', label='누적 설명된 분산 비율')
        ax2.set_ylabel('누적 설명된 분산 비율')
        ax2.tick_params(axis='y')
        
        # 일반적인 임계값 표시 (0.8, 0.9, 0.95)
        thresholds = [0.8, 0.9, 0.95]
        for threshold in thresholds:
            if max(variance_df['cumulative_variance_ratio']) >= threshold:
                n_for_threshold = next((i for i, ratio in enumerate(variance_df['cumulative_variance_ratio']) if ratio >= threshold), len(variance_df))
                ax2.axhline(y=threshold, color='gray', linestyle='--', alpha=0.5)
                ax2.text(max(variance_df['n_components'])*0.95, threshold, f'{threshold:.0%}', verticalalignment='bottom', horizontalalignment='right')
                ax2.axvline(x=n_for_threshold+1, color='gray', linestyle='--', alpha=0.5)
        
        # 타이틀 및 범례
        plt.title('주성분 개수에 따른 설명된 분산 비율')
        fig.tight_layout()
        
        # 범례 위치 조정
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
        
        return fig


class TSNE(DimensionalityReducer):
    """t-SNE를 위한 클래스"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Parameters:
        -----------
        df : DataFrame
            분석할 데이터프레임
        """
        super().__init__(df)
    
    @st.cache_data(ttl=3600)
    def perform_tsne(self, n_components: int = 2, columns: List[str] = None, 
                    scaler_type: str = 'standard', perplexity: float = 30.0,
                    learning_rate: float = 200.0, n_iter: int = 1000) -> pd.DataFrame:
        """t-SNE를 수행합니다.
        
        Parameters:
        -----------
        n_components : int, default=2
            출력 차원 수
        columns : list, default=None
            분석할 열 목록 (None이면 모든 수치형 열)
        scaler_type : str, default='standard'
            스케일러 유형 ('standard', 'robust', 'minmax')
        perplexity : float, default=30.0
            t-SNE 퍼플렉시티 매개변수
        learning_rate : float, default=200.0
            t-SNE 학습률
        n_iter : int, default=1000
            최대 반복 횟수
        
        Returns:
        --------
        DataFrame
            t-SNE 결과 데이터프레임
        """
        try:
            from sklearn.manifold import TSNE as SklearnTSNE
            
            # 시작 시간 기록
            start_time = time.time()
            
            # 분석할 열 선택
            if columns is None:
                columns = self.numeric_cols
            else:
                # 존재하지 않는 열 제외
                columns = [col for col in columns if col in self.df.columns]
                # 수치형이 아닌 열 제외
                columns = [col for col in columns if pd.api.types.is_numeric_dtype(self.df[col])]
            
            if not columns:
                raise ValueError("분석할 수치형 열이 없습니다.")
            
            # 결측치 처리
            data = self.df[columns].fillna(self.df[columns].median())
            
            # 데이터 스케일링
            scaled_data = self._scale_data(columns, scaler_type)
            
            # 대용량 데이터 감지 및 경고
            if len(data) > 10000:
                warnings.warn(f"t-SNE는 대용량 데이터({len(data)}행)에 시간이 오래 걸릴 수 있습니다. 샘플링을 고려하세요.")
                # 대용량 데이터 샘플링 (선택적)
                # scaled_data = scaled_data[np.random.choice(len(scaled_data), 10000, replace=False)]
            
            # t-SNE 수행
            tsne = SklearnTSNE(
                n_components=n_components,
                perplexity=min(perplexity, len(data) - 1),  # perplexity는 데이터 수보다 작아야 함
                learning_rate=learning_rate,
                n_iter=n_iter,
                random_state=42
            )
            self.reduced_data = tsne.fit_transform(scaled_data)
            self.reduction_model = tsne
            
            # t-SNE 결과를 데이터프레임으로 변환
            tsne_df = pd.DataFrame(
                data=self.reduced_data,
                columns=[f'TSNE{i+1}' for i in range(n_components)],
                index=data.index
            )
            
            # 범주형 열 추가
            categorical_cols = self.df.select_dtypes(exclude=['number']).columns
            if not categorical_cols.empty:
                for col in categorical_cols:
                    tsne_df[col] = self.df[col].values
            
            # 실행 시간 기록
            execution_time = time.time() - start_time
            if 'tsne' not in self.execution_times:
                self.execution_times['tsne'] = []
            self.execution_times['tsne'].append(execution_time)
            
            return tsne_df
            
        except ImportError:
            print("t-SNE를 수행하려면 scikit-learn이 필요합니다.")
            return pd.DataFrame()
        except Exception as e:
            print(f"t-SNE 수행 중 오류 발생: {str(e)}")
            return pd.DataFrame()


class UMAP(DimensionalityReducer):
    """UMAP을 위한 클래스"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Parameters:
        -----------
        df : DataFrame
            분석할 데이터프레임
        """
        super().__init__(df)
    
    @st.cache_data(ttl=3600)
    def perform_umap(self, n_components: int = 2, columns: List[str] = None, 
                   scaler_type: str = 'standard', n_neighbors: int = 15,
                   min_dist: float = 0.1, metric: str = 'euclidean') -> pd.DataFrame:
        """UMAP을 수행합니다.
        
        Parameters:
        -----------
        n_components : int, default=2
            출력 차원 수
        columns : list, default=None
            분석할 열 목록 (None이면 모든 수치형 열)
        scaler_type : str, default='standard'
            스케일러 유형 ('standard', 'robust', 'minmax')
        n_neighbors : int, default=15
            로컬 근방 크기
        min_dist : float, default=0.1
            임베딩 점들 사이의 최소 거리
        metric : str, default='euclidean'
            거리 측정 방식
        
        Returns:
        --------
        DataFrame
            UMAP 결과 데이터프레임
        """
        try:
            import umap
            
            # 시작 시간 기록
            start_time = time.time()
            
            # 분석할 열 선택
            if columns is None:
                columns = self.numeric_cols
            else:
                # 존재하지 않는 열 제외
                columns = [col for col in columns if col in self.df.columns]
                # 수치형이 아닌 열 제외
                columns = [col for col in columns if pd.api.types.is_numeric_dtype(self.df[col])]
            
            if not columns:
                raise ValueError("분석할 수치형 열이 없습니다.")
            
            # 결측치 처리
            data = self.df[columns].fillna(self.df[columns].median())
            
            # 데이터 스케일링
            scaled_data = self._scale_data(columns, scaler_type)
            
            # UMAP 수행
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric=metric,
                random_state=42
            )
            self.reduced_data = reducer.fit_transform(scaled_data)
            self.reduction_model = reducer
            
            # UMAP 결과를 데이터프레임으로 변환
            umap_df = pd.DataFrame(
                data=self.reduced_data,
                columns=[f'UMAP{i+1}' for i in range(n_components)],
                index=data.index
            )
            
            # 범주형 열 추가
            categorical_cols = self.df.select_dtypes(exclude=['number']).columns
            if not categorical_cols.empty:
                for col in categorical_cols:
                    umap_df[col] = self.df[col].values
            
            # 실행 시간 기록
            execution_time = time.time() - start_time
            if 'umap' not in self.execution_times:
                self.execution_times['umap'] = []
            self.execution_times['umap'].append(execution_time)
            
            return umap_df
            
        except ImportError:
            print("UMAP을 수행하려면 umap-learn 패키지가 필요합니다. pip install umap-learn")
            return pd.DataFrame()
        except Exception as e:
            print(f"UMAP 수행 중 오류 발생: {str(e)}")
            return pd.DataFrame()