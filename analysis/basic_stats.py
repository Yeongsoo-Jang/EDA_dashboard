# analysis/basic_stats.py - 기초 통계 분석 관련 클래스 및 함수
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import streamlit as st
from scipy import stats
import warnings
import time

class DataAnalyzer:
    """데이터 분석을 위한 기본 클래스"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Parameters:
        -----------
        df : DataFrame
            분석할 데이터프레임
        """
        self.df = df
        self.numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        self.datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
        
        # 성능 모니터링
        self.execution_times = {}
    
    def _time_it(self, func_name):
        """함수 실행 시간을 측정하는 데코레이터 역할을 하는 메서드"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                
                if func_name not in self.execution_times:
                    self.execution_times[func_name] = []
                
                self.execution_times[func_name].append(end_time - start_time)
                return result
            return wrapper
        return decorator
    
    def get_column_types(self) -> Dict[str, List[str]]:
        """데이터프레임의 열 유형을 분류합니다."""
        # 고유값 수가 적은 수치형 열은 범주형으로 재분류
        for col in self.numeric_cols.copy():
            if self.df[col].nunique() <= 10 and self.df[col].nunique() / len(self.df) < 0.05:
                self.categorical_cols.append(col)
                self.numeric_cols.remove(col)
        
        # 날짜 관련 키워드가 있지만 datetime64 타입이 아닌 열은 확인
        date_keywords = ['date', 'time', 'year', 'month', 'day', '일자', '날짜', '연도', '월', '일']
        for col in self.df.columns:
            if col not in self.datetime_cols and any(keyword in col.lower() for keyword in date_keywords):
                if col in self.categorical_cols:
                    try:
                        # 첫 번째 비결측 값으로 날짜 변환 시도
                        sample = self.df[col].dropna().iloc[0] if not self.df[col].dropna().empty else None
                        if sample and isinstance(sample, str):
                            pd.to_datetime(sample)
                            print(f"'{col}' 열은 날짜 형식일 수 있습니다. to_datetime()으로 변환을 고려하세요.")
                    except:
                        pass
        
        return {
            'numeric': self.numeric_cols,
            'categorical': self.categorical_cols,
            'datetime': self.datetime_cols
        }

    def get_memory_usage(self) -> Dict[str, float]:
        """데이터프레임의 메모리 사용량을 계산합니다."""
        memory_usage = self.df.memory_usage(deep=True)
        total_memory = memory_usage.sum()
        
        # 열별 메모리 사용량
        column_memory = {col: memory_usage[i] / (1024 * 1024) for i, col in enumerate(self.df.columns)}
        
        return {
            'total_mb': total_memory / (1024 * 1024),  # MB 단위
            'column_memory_mb': column_memory,
            'row_count': len(self.df),
            'column_count': len(self.df.columns)
        }


class BasicStatistics(DataAnalyzer):
    """기초 통계 분석을 위한 클래스"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Parameters:
        -----------
        df : DataFrame
            분석할 데이터프레임
        """
        super().__init__(df)
    
    @st.cache_data(ttl=3600)
    def get_basic_stats(self, percentiles: List[float] = None) -> pd.DataFrame:
        """수치형 변수의 기초 통계량을 계산합니다.
        
        Parameters:
        -----------
        percentiles : list, default=None
            계산할 백분위수 목록 (None이면 [0.25, 0.5, 0.75])
        
        Returns:
        --------
        DataFrame
            기초 통계량이 포함된 데이터프레임
        """
        if percentiles is None:
            percentiles = [0.25, 0.5, 0.75]
        
        numeric_df = self.df[self.numeric_cols]
        
        # 성능 최적화: 대용량 데이터인 경우 일부 통계만 계산
        is_large_dataset = len(self.df) > 100000
        
        # 기본 통계량 계산
        stats_dict = {
            '평균': numeric_df.mean(),
            '중앙값': numeric_df.median(),
            '최소값': numeric_df.min(),
            '최대값': numeric_df.max(),
            '표준편차': numeric_df.std(),
            '결측치 수': self.df[self.numeric_cols].isnull().sum()
        }
        
        # 백분위수 계산
        for p in percentiles:
            stats_dict[f'{int(p*100)}%'] = numeric_df.quantile(p)
        
        # 대용량 데이터가 아닌 경우 추가 통계 계산
        if not is_large_dataset:
            stats_dict.update({
                '왜도': numeric_df.skew(),
                '첨도': numeric_df.kurt(),
                '변동계수': numeric_df.std() / numeric_df.mean().replace(0, np.nan)
            })
            
            # 정규성 검정 (Shapiro-Wilk)
            normality_test = {}
            for col in self.numeric_cols:
                # 표본 크기가 너무 크면 검정이 항상 귀무가설을 기각하는 경향이 있음
                # 최대 5000개 샘플로 제한
                sample_size = min(5000, len(self.df))
                sample = self.df[col].dropna().sample(n=sample_size, random_state=42) if len(self.df[col].dropna()) > sample_size else self.df[col].dropna()
                
                if len(sample) >= 3:  # 최소 3개 이상의 데이터 필요
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            stat, p_value = stats.shapiro(sample)
                            normality_test[col] = p_value > 0.05  # True면 정규분포, False면 비정규분포
                    except:
                        normality_test[col] = None
                else:
                    normality_test[col] = None
            
            stats_dict['정규성'] = pd.Series(normality_test)
        
        # 결과 데이터프레임 생성
        stats_df = pd.DataFrame(stats_dict)
        
        return stats_df
    
    @st.cache_data(ttl=3600)
    def get_categorical_stats(self) -> pd.DataFrame:
        """범주형 변수의 통계를 계산합니다.
        
        Returns:
        --------
        DataFrame
            범주형 변수의 통계가 포함된 데이터프레임
        """
        categorical_df = self.df[self.categorical_cols]
        
        if categorical_df.empty:
            return pd.DataFrame()
        
        stats = {}
        
        for col in self.categorical_cols:
            value_counts = self.df[col].value_counts()
            unique_count = self.df[col].nunique()
            
            # 최빈값
            top_values = value_counts.head(3).index.tolist() if not value_counts.empty else []
            top_values_str = ", ".join([str(val) for val in top_values])
            
            # 범주별 비율
            total_count = value_counts.sum()
            top_value_percent = value_counts.iloc[0] / total_count * 100 if not value_counts.empty else 0
            
            stats[col] = {
                '고유값 수': unique_count,
                '최빈값': top_values_str,
                '최빈값 빈도': value_counts.iloc[0] if not value_counts.empty else 0,
                '최빈값 비율(%)': round(top_value_percent, 2),
                '결측치 수': self.df[col].isnull().sum(),
                '결측치 비율(%)': round(self.df[col].isnull().sum() / len(self.df) * 100, 2),
                '엔트로피': self._calculate_entropy(self.df[col])
            }
        
        return pd.DataFrame(stats).T
    
    def _calculate_entropy(self, series: pd.Series) -> float:
        """범주형 변수의 엔트로피를 계산합니다."""
        value_counts = series.value_counts(normalize=True)
        return -sum(p * np.log2(p) for p in value_counts if p > 0)
    
    @st.cache_data(ttl=3600)
    def get_data_summary(self) -> Dict[str, Any]:
        """데이터의 전반적인 요약 정보를 제공합니다.
        
        Returns:
        --------
        dict
            데이터 요약 정보가 포함된 딕셔너리
        """
        column_types = self.get_column_types()
        
        summary = {
            '행 수': len(self.df),
            '열 수': len(self.df.columns),
            '결측치가 있는 열 수': self.df.isna().any().sum(),
            '중복된 행 수': self.df.duplicated().sum(),
            '수치형 변수 수': len(column_types['numeric']),
            '범주형 변수 수': len(column_types['categorical']),
            '날짜형 변수 수': len(column_types['datetime']),
            '메모리 사용량(MB)': self.df.memory_usage(deep=True).sum() / (1024 * 1024)
        }
        
        # 대표적인 수치형/범주형 변수 통계
        if column_types['numeric']:
            numeric_sample = self.df[column_types['numeric']].agg(['mean', 'median', 'std']).head(5)
            summary['수치형 변수 통계 샘플'] = numeric_sample
        
        if column_types['categorical']:
            cat_sample = {}
            for col in column_types['categorical'][:5]:  # 처음 5개만
                value_counts = self.df[col].value_counts().head(3)
                cat_sample[col] = value_counts.to_dict()
            summary['범주형 변수 통계 샘플'] = cat_sample
        
        return summary
    
    def get_correlation_matrix(self, method: str = 'pearson', min_periods: int = None) -> pd.DataFrame:
        """수치형 변수 간의 상관관계 행렬을 계산합니다.
        
        Parameters:
        -----------
        method : str, default='pearson'
            상관계수 계산 방식 ('pearson', 'spearman', 'kendall')
        min_periods : int, default=None
            상관계수 계산에 필요한 최소 관측치 수
        
        Returns:
        --------
        DataFrame
            상관관계 행렬
        """
        # 수치형 열에 대해서만 상관관계 계산
        numeric_df = self.df[self.numeric_cols]
        
        # 대용량 데이터인 경우 샘플링
        is_large_dataset = len(self.df) > 100000
        if is_large_dataset:
            numeric_df = numeric_df.sample(n=100000, random_state=42)
        
        # 상관관계 계산
        corr_matrix = numeric_df.corr(method=method, min_periods=min_periods)
        
        return corr_matrix
    
    def get_highly_correlated_pairs(self, threshold: float = 0.7, method: str = 'pearson') -> List[Tuple[str, str, float]]:
        """높은 상관관계를 가진 변수 쌍을 찾습니다.
        
        Parameters:
        -----------
        threshold : float, default=0.7
            상관계수 임계값
        method : str, default='pearson'
            상관계수 계산 방식
        
        Returns:
        --------
        list
            높은 상관관계를 가진 변수 쌍 목록 [(변수1, 변수2, 상관계수)]
        """
        corr_matrix = self.get_correlation_matrix(method=method)
        
        # 상삼각 행렬만 고려 (중복 제거)
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) >= threshold:
                    var1 = corr_matrix.columns[i]
                    var2 = corr_matrix.columns[j]
                    corr_value = corr_matrix.iloc[i, j]
                    corr_pairs.append((var1, var2, corr_value))
        
        # 상관계수 절대값 기준 내림차순 정렬
        corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        return corr_pairs
    
    def describe_column(self, column: str) -> Dict[str, Any]:
        """특정 열에 대한 상세 통계 정보를 제공합니다.
        
        Parameters:
        -----------
        column : str
            분석할 열 이름
        
        Returns:
        --------
        dict
            열에 대한 상세 통계 정보
        """
        if column not in self.df.columns:
            return {"error": f"열 '{column}'이 데이터프레임에 존재하지 않습니다."}
        
        col_data = self.df[column]
        
        # 기본 정보
        result = {
            "name": column,
            "dtype": str(col_data.dtype),
            "count": len(col_data),
            "missing_count": col_data.isnull().sum(),
            "missing_percent": col_data.isnull().sum() / len(col_data) * 100,
            "unique_count": col_data.nunique()
        }
        
        # 데이터 유형별 추가 분석
        if pd.api.types.is_numeric_dtype(col_data):
            # 수치형 데이터 분석
            stats = col_data.describe()
            result.update({
                "min": stats['min'],
                "max": stats['max'],
                "mean": stats['mean'],
                "median": col_data.median(),
                "std": stats['std'],
                "skewness": col_data.skew(),
                "kurtosis": col_data.kurt(),
                "zeros_count": (col_data == 0).sum(),
                "negative_count": (col_data < 0).sum()
            })
            
            # 이상치 감지
            q1 = stats['25%']
            q3 = stats['75%']
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
            
            result.update({
                "iqr": iqr,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "outliers_count": len(outliers),
                "outliers_percent": len(outliers) / col_data.count() * 100
            })
            
        elif pd.api.types.is_categorical_dtype(col_data) or pd.api.types.is_object_dtype(col_data):
            # 범주형 데이터 분석
            value_counts = col_data.value_counts()
            top_categories = value_counts.head(5).to_dict()
            
            result.update({
                "top_categories": top_categories,
                "entropy": self._calculate_entropy(col_data),
                "is_binary": col_data.nunique() == 2
            })
            
            # 최빈값
            if not value_counts.empty:
                result.update({
                    "mode": value_counts.index[0],
                    "mode_count": value_counts.iloc[0],
                    "mode_percent": value_counts.iloc[0] / col_data.count() * 100
                })
            
        elif pd.api.types.is_datetime64_dtype(col_data):
            # 날짜/시간 데이터 분석
            valid_dates = col_data.dropna()
            
            if not valid_dates.empty:
                result.update({
                    "min_date": valid_dates.min(),
                    "max_date": valid_dates.max(),
                    "range_days": (valid_dates.max() - valid_dates.min()).days,
                    "weekday_distribution": valid_dates.dt.weekday.value_counts().to_dict(),
                    "month_distribution": valid_dates.dt.month.value_counts().to_dict()
                })
        
        return result


class AdvancedStatistics(BasicStatistics):
    """고급 통계 분석을 위한 클래스"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Parameters:
        -----------
        df : DataFrame
            분석할 데이터프레임
        """
        super().__init__(df)
    
    def get_variable_importance(self, target_column: str, method: str = 'correlation') -> pd.DataFrame:
        """타겟 변수에 대한 각 변수의 중요도를 계산합니다.
        
        Parameters:
        -----------
        target_column : str
            타겟 변수 열 이름
        method : str, default='correlation'
            중요도 계산 방식 ('correlation', 'mutual_info', 'rf_importance')
        
        Returns:
        --------
        DataFrame
            변수 중요도 데이터프레임
        """
        if target_column not in self.df.columns:
            return pd.DataFrame({"error": f"타겟 변수 '{target_column}'이 데이터프레임에 존재하지 않습니다."})
        
        # 타겟 변수가 수치형인지 확인
        if not pd.api.types.is_numeric_dtype(self.df[target_column]):
            if method == 'correlation':
                print(f"타겟 변수 '{target_column}'이 수치형이 아닙니다. 다른 방법을 사용하세요.")
                method = 'mutual_info'
        
        if method == 'correlation':
            # 상관계수 기반 중요도
            numeric_df = self.df[self.numeric_cols]
            # 타겟을 제외한 다른 변수들과의 상관관계 계산
            if target_column in numeric_df.columns:
                correlations = numeric_df.corr()[target_column].drop(target_column)
                importance_df = pd.DataFrame({'variable': correlations.index, 'importance': correlations.abs()})
                importance_df = importance_df.sort_values('importance', ascending=False)
                
                return importance_df
        
        elif method == 'mutual_info':
            # 상호 정보량 기반 중요도 (분류/회귀)
            from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
            
            # 대상 변수를 제외한 나머지 변수들
            X = self.df.drop(columns=[target_column])
            y = self.df[target_column]
            
            # 범주형 변수는 원-핫 인코딩
            X_encoded = pd.get_dummies(X, drop_first=True)
            
            # 타겟이 범주형인지 수치형인지에 따라 함수 선택
            if pd.api.types.is_numeric_dtype(y):
                mi_scores = mutual_info_regression(X_encoded, y)
            else:
                mi_scores = mutual_info_classif(X_encoded, y)
            
            # 인코딩된 변수명과 원래 변수명 매핑
            original_names = []
            for col in X_encoded.columns:
                # 원-핫 인코딩된 경우 원래 변수명 추출
                if '_' in col:
                    original_name = col.split('_')[0]
                    if original_name not in original_names:
                        original_names.append(original_name)
                else:
                    original_names.append(col)
            
            # 변수별 중요도 합산
            importance_dict = {}
            for idx, col in enumerate(X_encoded.columns):
                if '_' in col:
                    original_name = col.split('_')[0]
                    if original_name in importance_dict:
                        importance_dict[original_name] += mi_scores[idx]
                    else:
                        importance_dict[original_name] = mi_scores[idx]
                else:
                    importance_dict[col] = mi_scores[idx]
            
            importance_df = pd.DataFrame({
                'variable': list(importance_dict.keys()), 
                'importance': list(importance_dict.values())
            })
            importance_df = importance_df.sort_values('importance', ascending=False)
            
            return importance_df
        
        elif method == 'rf_importance':
            # 랜덤 포레스트 기반 중요도
            try:
                from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
                
                # 대상 변수를 제외한 나머지 변수들
                X = self.df.drop(columns=[target_column])
                y = self.df[target_column]
                
                # 범주형 변수는 원-핫 인코딩
                X_encoded = pd.get_dummies(X, drop_first=True)
                
                # 타겟이 범주형인지 수치형인지에 따라 모델 선택
                if pd.api.types.is_numeric_dtype(y):
                    model = RandomForestRegressor(n_estimators=50, random_state=42)
                else:
                    model = RandomForestClassifier(n_estimators=50, random_state=42)
                
                # 모델 학습
                model.fit(X_encoded, y)
                
                # 변수 중요도 추출
                importances = model.feature_importances_
                
                # 원본 변수별 중요도 합산 (원-핫 인코딩된 변수 처리)
                importance_dict = {}
                for idx, col in enumerate(X_encoded.columns):
                    if '_' in col:
                        original_name = col.split('_')[0]
                        if original_name in importance_dict:
                            importance_dict[original_name] += importances[idx]
                        else:
                            importance_dict[original_name] = importances[idx]
                    else:
                        importance_dict[col] = importances[idx]
                
                importance_df = pd.DataFrame({
                    'variable': list(importance_dict.keys()), 
                    'importance': list(importance_dict.values())
                })
                importance_df = importance_df.sort_values('importance', ascending=False)
                
                return importance_df
                
            except ImportError:
                print("변수 중요도 계산을 위해 scikit-learn이 필요합니다.")
                return pd.DataFrame()
        
        else:
            print(f"지원하지 않는 방법: {method}")
            return pd.DataFrame()
    
    def perform_distribution_test(self, column: str, test_type: str = 'normality') -> Dict[str, Any]:
        """분포 검정을 수행합니다.
        
        Parameters:
        -----------
        column : str
            검정할 열 이름
        test_type : str, default='normality'
            검정 유형 ('normality', 'uniformity')
        
        Returns:
        --------
        dict
            검정 결과
        """
        if column not in self.df.columns:
            return {"error": f"열 '{column}'이 데이터프레임에 존재하지 않습니다."}
        
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            return {"error": f"열 '{column}'이 수치형이 아닙니다."}
        
        # 결측치 제거
        data = self.df[column].dropna()
        
        # 표본 크기가 너무 크면 검정이 항상 귀무가설을 기각하는 경향이 있음
        # 최대 5000개 샘플로 제한
        sample_size = min(5000, len(data))
        sample = data.sample(n=sample_size, random_state=42) if len(data) > sample_size else data
        
        results = {
            "column": column,
            "sample_size": len(sample)
        }
        
        if test_type == 'normality':
            # 정규성 검정
            if len(sample) >= 3:  # 최소 3개 이상의 데이터 필요
                # Shapiro-Wilk 검정 (표본 크기가 작을 때)
                if len(sample) <= 5000:
                    try:
                        stat, p_value = stats.shapiro(sample)
                        results.update({
                            "test": "Shapiro-Wilk",
                            "statistic": stat,
                            "p_value": p_value,
                            "is_normal": p_value > 0.05
                        })
                    except:
                        pass
                
                # D'Agostino-Pearson's K^2 검정
                try:
                    stat, p_value = stats.normaltest(sample)
                    results.update({
                        "test": "D'Agostino-Pearson",
                        "statistic": stat,
                        "p_value": p_value,
                        "is_normal": p_value > 0.05
                    })
                except:
                    pass
                
                # 추가 정보
                results.update({
                    "skewness": sample.skew(),
                    "kurtosis": sample.kurt(),
                    "mean": sample.mean(),
                    "median": sample.median(),
                    "std": sample.std()
                })
            else:
                results["error"] = "검정을 위한 데이터가 충분하지 않습니다 (최소 3개 필요)."
        
        elif test_type == 'uniformity':
            # 균일 분포 검정 (Kolmogorov-Smirnov)
            if len(sample) >= 2:  # 최소 2개 이상의 데이터 필요
                try:
                    # 이론적 균일 분포
                    uniform_data = np.linspace(sample.min(), sample.max(), len(sample))
                    stat, p_value = stats.ks_2samp(sample, uniform_data)
                    
                    results.update({
                        "test": "Kolmogorov-Smirnov",
                        "statistic": stat,
                        "p_value": p_value,
                        "is_uniform": p_value > 0.05
                    })
                except:
                    pass
            else:
                results["error"] = "검정을 위한 데이터가 충분하지 않습니다 (최소 2개 필요)."
        
        return results
    
    def get_feature_interactions(self, features: List[str], target: str = None) -> pd.DataFrame:
        """특성 간의 상호작용을 분석합니다.
        
        Parameters:
        -----------
        features : list
            분석할 특성 목록
        target : str, default=None
            타겟 변수 (있는 경우)
        
        Returns:
        --------
        DataFrame
            상호작용 결과
        """
        # 데이터프레임에 없는 특성 확인
        missing_features = [f for f in features if f not in self.df.columns]
        if missing_features:
            return pd.DataFrame({"error": f"다음 특성이 데이터프레임에 없습니다: {missing_features}"})
        
        if target and target not in self.df.columns:
            return pd.DataFrame({"error": f"타겟 변수 '{target}'이 데이터프레임에 존재하지 않습니다."})
        
        results = []
        
        # 두 특성씩 조합
        for i in range(len(features)):
            for j in range(i+1, len(features)):
                feature1 = features[i]
                feature2 = features[j]
                
                # 두 특성이 모두 수치형인 경우
                if pd.api.types.is_numeric_dtype(self.df[feature1]) and pd.api.types.is_numeric_dtype(self.df[feature2]):
                    # 상관계수
                    correlation = self.df[[feature1, feature2]].corr().iloc[0, 1]
                    
                    interaction = {
                        'feature1': feature1,
                        'feature2': feature2,
                        'type': 'numeric-numeric',
                        'correlation': correlation
                    }
                    
                    # 타겟이 있고 수치형인 경우, 상호작용 항 생성 및 타겟과의 상관관계 계산
                    if target and pd.api.types.is_numeric_dtype(self.df[target]):
                        interaction_term = self.df[feature1] * self.df[feature2]
                        corr_with_target = np.corrcoef(interaction_term, self.df[target])[0, 1]
                        
                        interaction['interaction_corr_with_target'] = corr_with_target
                    
                    results.append(interaction)
                
                # 한 특성은 수치형, 다른 특성은 범주형인 경우
                elif (pd.api.types.is_numeric_dtype(self.df[feature1]) and 
                      (pd.api.types.is_categorical_dtype(self.df[feature2]) or pd.api.types.is_object_dtype(self.df[feature2]))):
                    # 범주별 평균, 분산 계산
                    group_stats = self.df.groupby(feature2)[feature1].agg(['mean', 'std']).reset_index()
                    
                    # 그룹 간 분산이 크면 강한 상호작용을 의미
                    group_means_variance = group_stats['mean'].var()
                    
                    interaction = {
                        'feature1': feature1,
                        'feature2': feature2,
                        'type': 'numeric-categorical',
                        'group_means_variance': group_means_variance
                    }
                    
                    results.append(interaction)
                
                elif (pd.api.types.is_numeric_dtype(self.df[feature2]) and 
                      (pd.api.types.is_categorical_dtype(self.df[feature1]) or pd.api.types.is_object_dtype(self.df[feature1]))):
                    # 범주별 평균, 분산 계산
                    group_stats = self.df.groupby(feature1)[feature2].agg(['mean', 'std']).reset_index()
                    
                    # 그룹 간 분산이 크면 강한 상호작용을 의미
                    group_means_variance = group_stats['mean'].var()
                    
                    interaction = {
                        'feature1': feature1,
                        'feature2': feature2,
                        'type': 'categorical-numeric',
                        'group_means_variance': group_means_variance
                    }
                    
                    results.append(interaction)
                
                # 두 특성이 모두 범주형인 경우
                elif ((pd.api.types.is_categorical_dtype(self.df[feature1]) or pd.api.types.is_object_dtype(self.df[feature1])) and
                      (pd.api.types.is_categorical_dtype(self.df[feature2]) or pd.api.types.is_object_dtype(self.df[feature2]))):
                    # 카이제곱 검정
                    try:
                        contingency_table = pd.crosstab(self.df[feature1], self.df[feature2])
                        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
                        
                        interaction = {
                            'feature1': feature1,
                            'feature2': feature2,
                            'type': 'categorical-categorical',
                            'chi2': chi2,
                            'p_value': p_value,
                            'dof': dof
                        }
                        
                        results.append(interaction)
                    except:
                        pass
        
        return pd.DataFrame(results)


class TextStatistics(DataAnalyzer):
    """텍스트 데이터 분석을 위한 클래스"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Parameters:
        -----------
        df : DataFrame
            분석할 데이터프레임
        """
        super().__init__(df)
        # 텍스트 열 식별 (object 타입이고 고유값이 많은 열)
        self.text_cols = []
        for col in self.categorical_cols:
            unique_ratio = self.df[col].nunique() / len(self.df)
            if unique_ratio > 0.5 and self.df[col].dtype == 'object':
                self.text_cols.append(col)
    
    def get_text_stats(self, column: str) -> Dict[str, Any]:
        """텍스트 열에 대한 기본 통계를 계산합니다.
        
        Parameters:
        -----------
        column : str
            분석할 텍스트 열 이름
        
        Returns:
        --------
        dict
            텍스트 통계 정보
        """
        if column not in self.df.columns:
            return {"error": f"열 '{column}'이 데이터프레임에 존재하지 않습니다."}
        
        if self.df[column].dtype != 'object':
            return {"error": f"열 '{column}'이 텍스트 데이터가 아닙니다."}
        
        # 결측치가 아닌 문자열 데이터만 사용
        text_data = self.df[column].dropna().astype(str)
        
        # 기본 통계
        stats = {
            "count": len(text_data),
            "missing_count": self.df[column].isnull().sum(),
            "unique_count": text_data.nunique(),
            "unique_ratio": text_data.nunique() / len(text_data) if len(text_data) > 0 else 0,
        }
        
        # 텍스트 길이 통계
        text_lengths = text_data.str.len()
        
        stats.update({
            "min_length": text_lengths.min(),
            "max_length": text_lengths.max(),
            "mean_length": text_lengths.mean(),
            "median_length": text_lengths.median()
        })
        
        # 단어 수 통계
        word_counts = text_data.str.split().str.len()
        
        stats.update({
            "min_words": word_counts.min(),
            "max_words": word_counts.max(),
            "mean_words": word_counts.mean(),
            "median_words": word_counts.median()
        })
        
        # 공백 및 숫자 비율
        stats.update({
            "empty_count": (text_data == "").sum(),
            "whitespace_only_count": text_data.str.isspace().sum(),
            "contains_numbers_count": text_data.str.contains(r'\d').sum(),
            "contains_numbers_ratio": text_data.str.contains(r'\d').mean()
        })
        
        return stats
    
    def get_word_frequencies(self, column: str, top_n: int = 20, min_freq: int = 2, 
                           exclude_stopwords: bool = True, language: str = 'english') -> pd.DataFrame:
        """텍스트 열에서 단어 빈도를 계산합니다.
        
        Parameters:
        -----------
        column : str
            분석할 텍스트 열 이름
        top_n : int, default=20
            반환할 상위 단어 수
        min_freq : int, default=2
            포함할 최소 출현 빈도
        exclude_stopwords : bool, default=True
            불용어 제외 여부
        language : str, default='english'
            언어 ('english', 'korean', ...)
        
        Returns:
        --------
        DataFrame
            단어 빈도 데이터프레임
        """
        if column not in self.df.columns:
            return pd.DataFrame({"error": f"열 '{column}'이 데이터프레임에 존재하지 않습니다."})
        
        try:
            import re
            from collections import Counter
            
            # 결측치가 아닌 문자열 데이터만 사용
            text_data = self.df[column].dropna().astype(str)
            
            # 모든 텍스트 결합
            all_text = " ".join(text_data)
            
            # 불용어 준비
            stopwords = []
            if exclude_stopwords:
                try:
                    if language == 'english':
                        from nltk.corpus import stopwords
                        stopwords = set(stopwords.words('english'))
                    elif language == 'korean':
                        # 한국어 기본 불용어 (예시)
                        stopwords = set(['이', '그', '저', '것', '이것', '저것', '그것', '및', '등', '등등'])
                except:
                    print("불용어 처리를 위해 nltk가 필요합니다.")
            
            # 단어 추출 (숫자, 특수문자 제거 후 소문자 변환)
            words = re.findall(r'\b[a-zA-Z가-힣]+\b', all_text.lower())
            
            # 불용어 제거
            if exclude_stopwords:
                words = [word for word in words if word not in stopwords]
            
            # 단어 빈도 계산
            word_counts = Counter(words)
            
            # 최소 빈도 이상인 단어만 선택
            word_counts = {word: count for word, count in word_counts.items() if count >= min_freq}
            
            # 상위 N개 단어 선택
            top_words = dict(sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:top_n])
            
            # 데이터프레임으로 변환
            result_df = pd.DataFrame({
                'word': list(top_words.keys()),
                'frequency': list(top_words.values())
            })
            
            return result_df
        
        except ImportError:
            print("텍스트 분석을 위해 NLTK가 필요할 수 있습니다.")
            return pd.DataFrame()