# utils/data_processor.py - 데이터 전처리 관련 함수
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from scipy import stats
import warnings
import re
from typing import Dict, List, Optional, Tuple, Union
import time
from config import INSIGHT_THRESHOLDS, DATA_TYPE_MAPPING

# 성능 모니터링을 위한 데코레이터
def time_it(func):
    """함수 실행 시간을 측정하는 데코레이터"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        
        if 'performance_logs' not in st.session_state:
            st.session_state['performance_logs'] = {}
        
        func_name = func.__name__
        if func_name not in st.session_state['performance_logs']:
            st.session_state['performance_logs'][func_name] = []
        
        st.session_state['performance_logs'][func_name].append(execution_time)
        
        # 대용량 데이터 처리 시 로그 남기기
        if execution_time > 1.0:  # 1초 이상 걸린 작업
            print(f"함수 {func_name} 실행 시간: {execution_time:.2f}초")
        
        return result
    
    return wrapper

@st.cache_data(ttl=3600, show_spinner=False)
def get_data_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """데이터프레임의 열 유형을 분류합니다."""
    # 기본 타입 분류
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    # 고유값 수가 적은 수치형 열은 범주형으로 재분류
    for col in numeric_cols.copy():
        if df[col].nunique() <= 10 and df[col].nunique() / len(df) < 0.05:
            categorical_cols.append(col)
            numeric_cols.remove(col)
    
    # 날짜 관련 키워드가 있지만 datetime64 타입이 아닌 열은 변환 시도
    date_keywords = ['date', 'time', 'year', 'month', 'day', '일자', '날짜', '연도', '월', '일']
    for col in df.columns:
        if col not in datetime_cols and any(keyword in col.lower() for keyword in date_keywords):
            if col in categorical_cols:
                try:
                    test_conversion = pd.to_datetime(df[col].iloc[0])
                    datetime_cols.append(col)
                    categorical_cols.remove(col)
                except:
                    pass
    
    return {
        'numeric': numeric_cols,
        'categorical': categorical_cols,
        'datetime': datetime_cols
    }

@time_it
def get_data_quality_report(df: pd.DataFrame) -> Dict:
    """데이터 품질 보고서를 생성합니다."""
    # 기본 정보 수집
    quality_report = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "total_cells": len(df) * len(df.columns),
        "missing_cells": df.isna().sum().sum(),
        "duplicate_rows": df.duplicated().sum(),
        "memory_usage": df.memory_usage(deep=True).sum() / (1024 * 1024),  # MB 단위
    }
    
    # 결측치 비율 계산
    quality_report["missing_percentage"] = (quality_report["missing_cells"] / quality_report["total_cells"]) * 100
    
    # 데이터 유형 분류
    data_types = get_data_types(df)
    quality_report["data_types"] = {
        "numeric_count": len(data_types["numeric"]),
        "categorical_count": len(data_types["categorical"]),
        "datetime_count": len(data_types["datetime"])
    }
    
    # 열별 품질 보고서 (대용량 데이터의 경우 일부 계산 생략)
    is_large_dataset = len(df) * len(df.columns) > 1000000  # 행*열이 백만 이상이면 대용량으로 간주
    
    columns_report = []
    for col in df.columns:
        col_info = {
            "column_name": col,
            "data_type": str(df[col].dtype),
            "unique_values": df[col].nunique(),
            "unique_percentage": (df[col].nunique() / len(df)) * 100 if len(df) > 0 else 0,
            "missing_count": df[col].isna().sum(),
            "missing_percentage": (df[col].isna().sum() / len(df)) * 100 if len(df) > 0 else 0,
        }
        
        # 데이터 유형별 추가 정보 (대용량 데이터 처리 최적화)
        if pd.api.types.is_numeric_dtype(df[col]):
            # 기본 통계 정보 (모든 데이터셋에 계산)
            col_info.update({
                "min": df[col].min() if not df[col].isna().all() else None,
                "max": df[col].max() if not df[col].isna().all() else None,
                "mean": df[col].mean() if not df[col].isna().all() else None,
                "median": df[col].median() if not df[col].isna().all() else None,
                "std": df[col].std() if not df[col].isna().all() else None,
            })
            
            # 계산 비용이 높은 작업은 대용량 데이터셋에서 생략
            if not is_large_dataset:
                if not df[col].isna().all():
                    col_info.update({
                        "zeros_count": (df[col] == 0).sum(),
                        "zeros_percentage": ((df[col] == 0).sum() / df[col].count()) * 100 if df[col].count() > 0 else 0,
                        "negative_count": (df[col] < 0).sum(),
                        "skewness": df[col].skew(),
                        "kurtosis": df[col].kurt(),
                    })
                    
                    # 이상치 감지 (IQR 방식) - 숫자형이지만 이진(0/1) 값만 갖는 경우 IQR 계산이 무의미하거나 오류를 유발할 수 있음
                    is_binary_numeric = False
                    if df[col].nunique(dropna=True) <= 2:
                        unique_values = df[col].dropna().unique()
                        if len(unique_values) > 0 and all(val in [0, 1, 0.0, 1.0] for val in unique_values):
                            is_binary_numeric = True

                    if not is_binary_numeric:
                        q1 = df[col].quantile(0.25)
                        q3 = df[col].quantile(0.75)
                        
                        if pd.notna(q1) and pd.notna(q3):
                            iqr = q3 - q1
                            lower_bound = q1 - 1.5 * iqr
                            upper_bound = q3 + 1.5 * iqr
                            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                            col_info["outliers_count"] = len(outliers)
                            col_info["outliers_percentage"] = (len(outliers) / df[col].count()) * 100 if df[col].count() > 0 else 0
                            col_info["outliers_lower_bound"] = lower_bound
                            col_info["outliers_upper_bound"] = upper_bound
                        else: # q1 또는 q3가 NaN인 경우 (예: 모든 값이 NaN)
                            col_info["outliers_count"] = 0
                            col_info["outliers_percentage"] = 0.0
                            col_info["outliers_lower_bound"] = np.nan
                            col_info["outliers_upper_bound"] = np.nan
                    else: # 이진 숫자형 열의 경우 IQR 기반 이상치 정보 N/A 처리
                        col_info["outliers_count"] = "N/A (binary numeric)"
                        col_info["outliers_percentage"] = "N/A (binary numeric)"
        
        elif pd.api.types.is_datetime64_dtype(df[col]):
            if not df[col].isna().all():
                col_info.update({
                    "min_date": df[col].min(),
                    "max_date": df[col].max(),
                    "range_days": (df[col].max() - df[col].min()).days if not df[col].isna().all() else None,
                })
        
        else:  # 범주형/문자열 데이터
            if not df[col].isna().all() and not is_large_dataset:
                # 최빈값과 그 빈도
                most_common = df[col].value_counts().nlargest(1)
                if not most_common.empty:
                    col_info.update({
                        "most_common_value": most_common.index[0],
                        "most_common_count": most_common.iloc[0],
                        "most_common_percentage": (most_common.iloc[0] / df[col].count()) * 100 if df[col].count() > 0 else 0,
                    })
                
                # 빈 문자열 수 (문자열 열인 경우만)
                if df[col].dtype == 'object':
                    empty_strings = (df[col] == "").sum()
                    col_info["empty_strings"] = empty_strings
                    col_info["empty_strings_percentage"] = (empty_strings / df[col].count()) * 100 if df[col].count() > 0 else 0
        
        columns_report.append(col_info)
    
    quality_report["columns"] = columns_report
    
    # 데이터 품질 점수 계산
    quality_score = calculate_data_quality_score(quality_report)
    quality_report["quality_score"] = quality_score
    
    return quality_report

def calculate_data_quality_score(quality_report: Dict) -> Dict:
    """데이터 품질 점수를 계산합니다."""
    # 기본 점수 (100점 만점)
    score = 100
    
    # 결측치에 따른 감점 (최대 -30점)
    missing_penalty = min(30, quality_report["missing_percentage"] * 3)
    score -= missing_penalty
    
    # 중복 행에 따른 감점 (최대 -20점)
    duplicate_percentage = (quality_report["duplicate_rows"] / quality_report["row_count"]) * 100 if quality_report["row_count"] > 0 else 0
    duplicate_penalty = min(20, duplicate_percentage * 4)
    score -= duplicate_penalty
    
    # 이상치에 따른 감점 (최대 -30점)
    outlier_penalty = 0
    for col_info in quality_report["columns"]:
        # "outliers_percentage"가 숫자인 경우에만 비교 및 감점 계산
        if "outliers_percentage" in col_info and \
           isinstance(col_info["outliers_percentage"], (int, float)) and \
           col_info["outliers_percentage"] > 5:
            outlier_penalty += min(10, col_info["outliers_percentage"] * 0.5) # type: ignore
    
    outlier_penalty = min(30, outlier_penalty)
    score -= outlier_penalty
    
    # 품질 등급 결정
    if score >= 90:
        grade = "A+"
        color = "#2C8D80"
    elif score >= 80:
        grade = "A"
        color = "#3DBFAD"
    elif score >= 70:
        grade = "B+"
        color = "#50E3C2"
    elif score >= 60:
        grade = "B"
        color = "#66D9E8"
    elif score >= 50:
        grade = "C+"
        color = "#FFD43B"
    elif score >= 40:
        grade = "C"
        color = "#FF9F1C"
    else:
        grade = "D"
        color = "#FF6B6B"
    
    return {
        "score": round(score, 1),
        "grade": grade,
        "color": color,
        "penalties": {
            "missing": round(missing_penalty, 1),
            "duplicates": round(duplicate_penalty, 1),
            "outliers": round(outlier_penalty, 1)
        }
    }

@time_it
def detect_outliers(df: pd.DataFrame, method: str = 'iqr', threshold: float = 1.5, columns: List[str] = None) -> Dict:
    """이상치를 감지합니다.
    
    Parameters:
    -----------
    df : DataFrame
        분석할 데이터프레임
    method : str, default='iqr'
        이상치 감지 방법 ('iqr', 'zscore', 'quantile')
    threshold : float, default=1.5
        이상치 감지 임계값 (IQR 방식의 경우 1.5가 일반적)
    columns : list, default=None
        이상치를 감지할 열 목록. None이면 모든 수치형 열을 분석
    
    Returns:
    --------
    dict
        열별 이상치 정보를 포함하는 딕셔너리
    """
    # 분석할 열 선택
    if columns is None:
        numeric_df = df.select_dtypes(include=['number'])
    else:
        numeric_df = df[columns].select_dtypes(include=['number'])
    
    # 대용량 데이터 처리를 위한 샘플링
    is_large_dataset = len(df) > 100000
    sample_df = numeric_df.sample(n=min(100000, len(numeric_df))) if is_large_dataset else numeric_df
    
    outliers = {}
    
    if method == 'iqr':  # 사분위수 범위 기반
        for column in numeric_df.columns:
            # 기본 통계 계산 (샘플링 데이터 사용)
            Q1 = sample_df[column].quantile(0.25)
            Q3 = sample_df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            # 실제 이상치 계산 (전체 데이터 사용)
            outlier_mask = (numeric_df[column] < lower_bound) | (numeric_df[column] > upper_bound)
            outlier_indices = numeric_df[outlier_mask].index.tolist()
            
            outliers[column] = {
                'count': len(outlier_indices),
                'percentage': len(outlier_indices) / len(numeric_df) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'indices': outlier_indices[:100] if len(outlier_indices) > 100 else outlier_indices  # 최대 100개 인덱스만 저장
            }
    
    elif method == 'zscore':  # Z-score 기반
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for column in numeric_df.columns:
                # 결측치 제거
                column_data = numeric_df[column].dropna()
                
                # Z-score 계산
                z_scores = np.abs(stats.zscore(column_data, nan_policy='omit'))
                
                # 이상치 식별
                outlier_mask = z_scores > threshold
                outlier_indices = column_data.index[outlier_mask].tolist()
                
                outliers[column] = {
                    'count': len(outlier_indices),
                    'percentage': len(outlier_indices) / len(column_data) * 100 if len(column_data) > 0 else 0,
                    'threshold': threshold,
                    'method': 'zscore',
                    'indices': outlier_indices[:100] if len(outlier_indices) > 100 else outlier_indices
                }
    
    elif method == 'quantile':  # 분위수 기반 (상/하위 n% 제외)
        lower_quantile = threshold / 100
        upper_quantile = 1 - lower_quantile
        
        for column in numeric_df.columns:
            lower_bound = numeric_df[column].quantile(lower_quantile)
            upper_bound = numeric_df[column].quantile(upper_quantile)
            
            outlier_mask = (numeric_df[column] < lower_bound) | (numeric_df[column] > upper_bound)
            outlier_indices = numeric_df[outlier_mask].index.tolist()
            
            outliers[column] = {
                'count': len(outlier_indices),
                'percentage': len(outlier_indices) / len(numeric_df) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'method': 'quantile',
                'quantile_threshold': threshold,
                'indices': outlier_indices[:100] if len(outlier_indices) > 100 else outlier_indices
            }
    
    return outliers

@time_it
def handle_missing_values(df: pd.DataFrame, strategy: str = 'smart', columns: Dict[str, str] = None) -> pd.DataFrame:
    """결측치를 처리합니다.
    
    Parameters:
    -----------
    df : DataFrame
        처리할 데이터프레임
    strategy : str, default='smart'
        결측치 처리 전략 ('smart', 'median', 'mean', 'zero', 'mode', 'drop', 'knn')
    columns : dict, default=None
        각 열별 처리 전략을 지정하는 딕셔너리 {'column_name': 'strategy'}
    
    Returns:
    --------
    DataFrame
        결측치가 처리된 데이터프레임
    """
    df_processed = df.copy()
    
    # 대용량 데이터 처리를 위한 최적화
    is_large_dataset = len(df) > 100000
    
    # 열별 처리 전략이 지정된 경우
    if columns is not None:
        for col, col_strategy in columns.items():
            if col in df.columns:
                df_processed[col] = handle_column_missing(df_processed, col, col_strategy, is_large_dataset)
        return df_processed
    
    # 스마트 전략 (데이터 특성에 맞게 자동 선택)
    if strategy == 'smart':
        for col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count == 0:
                continue
                
            # 결측치가 80% 이상인 경우 경고만 표시 (삭제는 선택적)
            if missing_count / len(df) > 0.8:
                print(f"Warning: '{col}' 열의 결측치가 80% 이상입니다. 이 열을 제거하는 것이 좋습니다.")
                continue
                
            # 데이터 유형에 따른 처리
            df_processed[col] = handle_column_missing(df_processed, col, 'auto', is_large_dataset)
    
    # 단일 전략 적용 (모든 열에 동일한 방식)
    else:
        for col in df.columns:
            if df[col].isna().sum() > 0:  # 결측치가 있는 열만 처리
                df_processed[col] = handle_column_missing(df_processed, col, strategy, is_large_dataset)
    
    return df_processed

def handle_column_missing(df: pd.DataFrame, column: str, strategy: str, is_large_dataset: bool) -> pd.Series:
    """단일 열의 결측치를 처리합니다."""
    if df[column].isna().sum() == 0:
        return df[column]
    
    # 자동 전략 (데이터 유형에 따라 최적의 방법 선택)
    if strategy == 'auto':
        if pd.api.types.is_numeric_dtype(df[column]):
            # 수치형 데이터: 분포에 따라 평균, 중앙값 또는 0으로 대체
            if is_large_dataset or df[column].skew() > 1 or df[column].skew() < -1:  # 치우친 분포 또는 대용량
                return df[column].fillna(df[column].median())
            else:  # 정규 분포에 가까움
                return df[column].fillna(df[column].mean())
        
        elif pd.api.types.is_datetime64_dtype(df[column]):
            # 날짜형 데이터: 가능하면 보간법 사용
            try:
                return df[column].interpolate(method='time')
            except:
                # 시간 보간이 실패하면 앞/뒤 값으로 채우기
                return df[column].fillna(method='ffill').fillna(method='bfill')
        
        else:
            # 범주형/문자열 데이터: 최빈값 또는 'Unknown'으로 대체
            mode_value = df[column].mode().iloc[0] if not df[column].mode().empty else 'Unknown'
            return df[column].fillna(mode_value)
    
    # 명시적 전략
    elif strategy == 'median':
        if pd.api.types.is_numeric_dtype(df[column]):
            return df[column].fillna(df[column].median())
        else:
            # 수치형이 아닌 열에는 적용 불가
            print(f"Warning: 중앙값 전략은 수치형 열에만 적용 가능합니다. '{column}' 열은 처리되지 않았습니다.")
            return df[column]
    
    elif strategy == 'mean':
        if pd.api.types.is_numeric_dtype(df[column]):
            return df[column].fillna(df[column].mean())
        else:
            print(f"Warning: 평균값 전략은 수치형 열에만 적용 가능합니다. '{column}' 열은 처리되지 않았습니다.")
            return df[column]
    
    elif strategy == 'zero':
        if pd.api.types.is_numeric_dtype(df[column]):
            return df[column].fillna(0)
        else:
            print(f"Warning: 0 대체 전략은 수치형 열에만 적용 가능합니다. '{column}' 열은 처리되지 않았습니다.")
            return df[column]
    
    elif strategy == 'mode':
        # 모든 데이터 유형에 적용 가능
        mode_value = df[column].mode().iloc[0] if not df[column].mode().empty else None
        if mode_value is not None:
            return df[column].fillna(mode_value)
        else:
            # 최빈값이 없으면 Unknown
            return df[column].fillna('Unknown')
    
    elif strategy == 'knn':
        # KNN 기반 결측치 대체 (대용량 데이터에서는 사용하지 않음)
        if is_large_dataset:
            print(f"Warning: KNN 전략은 대용량 데이터에 적용하기 어렵습니다. '{column}' 열은 중앙값으로 대체됩니다.")
            return df[column].fillna(df[column].median() if pd.api.types.is_numeric_dtype(df[column]) else df[column].mode().iloc[0])
        
        try:
            from sklearn.impute import KNNImputer
            # 단일 열만 처리하는 경우에도 KNN은 다변량 접근이 필요하므로 전체 수치형 데이터 사용
            numeric_cols = df.select_dtypes(include=['number']).columns
            if column in numeric_cols:
                imputer = KNNImputer(n_neighbors=5)
                imputed_data = imputer.fit_transform(df[numeric_cols])
                imputed_df = pd.DataFrame(imputed_data, columns=numeric_cols, index=df.index)
                return imputed_df[column]
            else:
                print(f"Warning: KNN 전략은 수치형 열에만 적용 가능합니다. '{column}' 열은 처리되지 않았습니다.")
                return df[column]
        except ImportError:
            print("Warning: KNN 결측치 처리를 위해서는 scikit-learn이 필요합니다.")
            return df[column]
    
    else:
        # 기본값: 결측치 그대로 유지
        return df[column]

@time_it
def scale_features(df: pd.DataFrame, method: str = 'standard', columns: List[str] = None) -> Tuple[pd.DataFrame, object]:
    """특성을 스케일링합니다.
    
    Parameters:
    -----------
    df : DataFrame
        스케일링할 데이터프레임
    method : str, default='standard'
        스케일링 방법 ('standard', 'minmax', 'robust')
    columns : list, default=None
        스케일링할 열 목록. None이면 모든 수치형 열 스케일링
    
    Returns:
    --------
    tuple
        (스케일링된 데이터프레임, 스케일러 객체)
    """
    if columns is None:
        numeric_df = df.select_dtypes(include=['number'])
        columns = numeric_df.columns
    else:
        numeric_df = df[columns]
    
    scaled_df = df.copy()
    
    if method == 'standard':
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)
    elif method == 'minmax':
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(numeric_df)
    elif method == 'robust':
        scaler = RobustScaler()
        scaled_data = scaler.fit_transform(numeric_df)
    else:
        raise ValueError(f"지원하지 않는 스케일링 방법: {method}")
    
    scaled_df[columns] = scaled_data
    
    return scaled_df, scaler

@time_it
def preprocess_data(df: pd.DataFrame, options: Dict = None) -> pd.DataFrame:
    """데이터를 전처리합니다.
    
    Parameters:
    -----------
    df : DataFrame
        전처리할 데이터프레임
    options : dict, default=None
        전처리 옵션 {'handle_missing': True, 'remove_duplicates': True, ...}
    
    Returns:
    --------
    DataFrame
        전처리된 데이터프레임
    """
    if options is None:
        options = {
            "handle_missing": True,
            "remove_duplicates": True,
            "normalize_columns": True,
            "convert_dates": True,
            "remove_outliers": False,
            "handle_low_variance": True
        }
    
    df_processed = df.copy()
    
    # 열 이름 정규화
    if options.get("normalize_columns", True):
        df_processed.columns = [normalize_column_name(col) for col in df_processed.columns]
    
    # 중복 행 제거
    if options.get("remove_duplicates", True):
        duplicate_count = df_processed.duplicated().sum()
        if duplicate_count > 0:
            df_processed = df_processed.drop_duplicates()
            print(f"중복 행 {duplicate_count}개가 제거되었습니다.")
    
    # 날짜 열 변환
    if options.get("convert_dates", True):
        for col in df_processed.columns:
            if df_processed[col].dtype == 'object':
                # 날짜형 열 자동 감지 및 변환
                if is_date_column(col, df_processed[col]):
                    try:
                        df_processed[col] = pd.to_datetime(df_processed[col], errors='coerce')
                        print(f"'{col}' 열이 날짜형으로 변환되었습니다.")
                    except:
                        pass
    
    # 범주형 데이터 인코딩
    if options.get("encode_categorical", False):
        categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            # 고유값이 적은 경우에만 원-핫 인코딩 적용
            if df_processed[col].nunique() < 10:
                dummies = pd.get_dummies(df_processed[col], prefix=col, drop_first=False)
                df_processed = pd.concat([df_processed, dummies], axis=1)
                df_processed.drop(col, axis=1, inplace=True)
                print(f"'{col}' 열이 원-핫 인코딩되었습니다.")
    
    # 결측치 처리
    if options.get("handle_missing", True):
        missing_columns = df_processed.columns[df_processed.isna().any()].tolist()
        if missing_columns:
            df_processed = handle_missing_values(df_processed, 'smart')
            print(f"{len(missing_columns)}개 열의 결측치가 처리되었습니다.")
    
    # 이상치 제거
    if options.get("remove_outliers", False):
        outliers = detect_outliers(df_processed)
        removed_count = 0
        
        for col, outlier_info in outliers.items():
            if outlier_info['percentage'] > 5:  # 이상치가 5% 이상인 경우에만 처리
                outlier_indices = outlier_info['indices']
                df_processed = df_processed.drop(index=outlier_indices)
                removed_count += len(outlier_indices)
        
        if removed_count > 0:
            print(f"이상치가 포함된 {removed_count}개 행이 제거되었습니다.")
    
    # 분산이 낮은 특성 제거
    if options.get("handle_low_variance", True):
        numeric_cols = df_processed.select_dtypes(include=['number']).columns
        low_variance_cols = []
        
        for col in numeric_cols:
            if df_processed[col].nunique() == 1:  # 모든 값이 동일
                low_variance_cols.append(col)
        
        if low_variance_cols:
            df_processed = df_processed.drop(columns=low_variance_cols)
            print(f"분산이 없는 {len(low_variance_cols)}개 열이 제거되었습니다: {', '.join(low_variance_cols)}")
    
    return df_processed

def normalize_column_name(column_name: str) -> str:
    """열 이름을 정규화합니다 (공백 제거, 소문자 변환, 특수 문자 대체)."""
    # 공백을 밑줄로 변환
    col = column_name.strip().lower().replace(' ', '_')
    
    # 특수 문자 제거/대체
    col = re.sub(r'[^\w\s]', '_', col)
    
    # 연속된 밑줄 제거
    col = re.sub(r'_+', '_', col)
    
    # 앞뒤 밑줄 제거
    col = col.strip('_')
    
    return col

def is_date_column(column_name: str, series: pd.Series) -> bool:
    """열이 날짜를 포함하는지 감지합니다."""
    # 열 이름에 날짜 관련 키워드가 있는지 확인
    date_keywords = ['date', 'time', 'year', 'month', 'day', '일자', '날짜', '연도', '월', '일']
    name_suggests_date = any(keyword in column_name.lower() for keyword in date_keywords)
    
    # 데이터 샘플이 날짜 형식인지 확인
    if series.dtype == 'object':
        # 빈 값이 아닌 첫 샘플 가져오기
        samples = series.dropna().head(5).tolist()
        if samples:
            date_patterns = [
                r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',  # YYYY-MM-DD or YYYY/MM/DD
                r'\d{1,2}[-/]\d{1,2}[-/]\d{4}',  # MM-DD-YYYY or DD-MM-YYYY
                r'\d{1,2}[-/]\d{1,2}[-/]\d{2}',  # MM-DD-YY or DD-MM-YY
                r'\d{4}\d{2}\d{2}',              # YYYYMMDD
                r'\d{4}년\s*\d{1,2}월\s*\d{1,2}일'  # 한국어 날짜 형식
            ]
            
            for sample in samples:
                if isinstance(sample, str):
                    # 정규식 패턴 매칭
                    if any(re.search(pattern, sample) for pattern in date_patterns):
                        return True
                    
                    # pd.to_datetime 시도
                    try:
                        pd.to_datetime(sample)
                        return True
                    except:
                        pass
    
    # 열 이름만으로 판단 (데이터 샘플이 날짜 형식이 아닌 경우)
    return name_suggests_date and series.dtype == 'object'

def encode_categorical_features(df: pd.DataFrame, method: str = 'onehot', max_categories: int = 10) -> pd.DataFrame:
    """범주형 특성을 인코딩합니다.
    
    Parameters:
    -----------
    df : DataFrame
        인코딩할 데이터프레임
    method : str, default='onehot'
        인코딩 방법 ('onehot', 'label', 'target', 'binary', 'frequency')
    max_categories : int, default=10
        원-핫 인코딩을 적용할 최대 고유값 수
    
    Returns:
    --------
    DataFrame
        인코딩된 데이터프레임
    """
    df_encoded = df.copy()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    for col in categorical_cols:
        unique_count = df[col].nunique()
        
        # 원-핫 인코딩 (고유값이 적은 경우)
        if method == 'onehot' and unique_count <= max_categories:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
            df_encoded = pd.concat([df_encoded.drop(col, axis=1), dummies], axis=1)
        
        # 레이블 인코딩
        elif method == 'label':
            df_encoded[col] = df[col].astype('category').cat.codes
        
        # 빈도 인코딩
        elif method == 'frequency':
            frequency_map = df[col].value_counts(normalize=True)
            df_encoded[col] = df[col].map(frequency_map)
        
        # 이진 인코딩 (고유값이 많은 경우에 효율적)
        elif method == 'binary' and unique_count > max_categories:
            from category_encoders import BinaryEncoder
            encoder = BinaryEncoder(cols=[col])
            temp_df = encoder.fit_transform(df[col])
            df_encoded = pd.concat([df_encoded.drop(col, axis=1), temp_df], axis=1)
    
    return df_encoded

# 추가: 데이터 밸런싱 함수
def balance_dataset(df: pd.DataFrame, target_column: str, method: str = 'undersample') -> pd.DataFrame:
    """분류 문제에서 데이터셋의 클래스 불균형을 조정합니다.
    
    Parameters:
    -----------
    df : DataFrame
        밸런싱할 데이터프레임
    target_column : str
        타겟 변수 (클래스) 열 이름
    method : str, default='undersample'
        밸런싱 방법 ('undersample', 'oversample', 'smote')
    
    Returns:
    --------
    DataFrame
        밸런싱된 데이터프레임
    """
    # 클래스 분포 확인
    class_counts = df[target_column].value_counts()
    min_class_count = class_counts.min()
    max_class_count = class_counts.max()
    imbalance_ratio = max_class_count / min_class_count
    
    # 불균형이 크지 않으면 원본 반환
    if imbalance_ratio < 1.5:
        print(f"클래스 불균형 비율이 낮습니다 ({imbalance_ratio:.2f}). 밸런싱이 필요하지 않습니다.")
        return df
    
    try:
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        
        # 언더샘플링
        if method == 'undersample':
            from imblearn.under_sampling import RandomUnderSampler
            sampler = RandomUnderSampler(random_state=42)
            X_resampled, y_resampled = sampler.fit_resample(X, y)
        
        # 오버샘플링
        elif method == 'oversample':
            from imblearn.over_sampling import RandomOverSampler
            sampler = RandomOverSampler(random_state=42)
            X_resampled, y_resampled = sampler.fit_resample(X, y)
        
        # SMOTE
        elif method == 'smote':
            try:
                from imblearn.over_sampling import SMOTE
                sampler = SMOTE(random_state=42)
                X_resampled, y_resampled = sampler.fit_resample(X, y)
            except ValueError:
                # SMOTE에 필요한 최소 샘플이 부족한 경우
                print("SMOTE를 적용할 수 없습니다. 대신 오버샘플링을 적용합니다.")
                from imblearn.over_sampling import RandomOverSampler
                sampler = RandomOverSampler(random_state=42)
                X_resampled, y_resampled = sampler.fit_resample(X, y)
        
        else:
            raise ValueError(f"지원하지 않는 밸런싱 방법: {method}")
        
        # 결과 병합
        df_balanced = pd.concat([X_resampled, y_resampled], axis=1)
        
        print(f"밸런싱 전: {dict(class_counts)}")
        print(f"밸런싱 후: {dict(df_balanced[target_column].value_counts())}")
        
        return df_balanced
    
    except ImportError:
        print("밸런싱을 위해 imbalanced-learn 패키지가 필요합니다. pip install imbalanced-learn")
        return df