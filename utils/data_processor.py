# utils/data_processor.py - 데이터 전처리 관련 함수
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import streamlit as st

def get_data_types(df):
    """데이터프레임의 열 유형을 분류합니다."""
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    return {
        'numeric': numeric_cols,
        'categorical': categorical_cols,
        'datetime': datetime_cols
    }

def get_data_quality_report(df):
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
    
    # 열별 품질 보고서
    columns_report = []
    for col in df.columns:
        col_info = {
            "column_name": col,
            "data_type": str(df[col].dtype),
            "unique_values": df[col].nunique(),
            "missing_count": df[col].isna().sum(),
            "missing_percentage": (df[col].isna().sum() / len(df)) * 100,
        }
        
        # 데이터 유형별 추가 정보
        if pd.api.types.is_numeric_dtype(df[col]):
            col_info.update({
                "min": df[col].min() if not df[col].isna().all() else None,
                "max": df[col].max() if not df[col].isna().all() else None,
                "mean": df[col].mean() if not df[col].isna().all() else None,
                "median": df[col].median() if not df[col].isna().all() else None,
                "std": df[col].std() if not df[col].isna().all() else None,
                "zeros_count": (df[col] == 0).sum(),
                "zeros_percentage": ((df[col] == 0).sum() / df[col].count()) * 100,
                "negative_count": (df[col] < 0).sum() if not df[col].isna().all() else 0,
            })
            
            # 이상치 감지 (IQR 방식)
            if not df[col].isna().all():
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                col_info["outliers_count"] = len(outliers)
                col_info["outliers_percentage"] = (len(outliers) / df[col].count()) * 100
        
        elif pd.api.types.is_datetime64_dtype(df[col]):
            col_info.update({
                "min_date": df[col].min() if not df[col].isna().all() else None,
                "max_date": df[col].max() if not df[col].isna().all() else None,
                "range_days": (df[col].max() - df[col].min()).days if not df[col].isna().all() else None,
            })
        
        else:  # 범주형/문자열 데이터
            if not df[col].isna().all():
                # 최빈값과 그 빈도
                most_common = df[col].value_counts().nlargest(1)
                col_info.update({
                    "most_common_value": most_common.index[0] if not most_common.empty else None,
                    "most_common_count": most_common.iloc[0] if not most_common.empty else None,
                    "most_common_percentage": (most_common.iloc[0] / df[col].count()) * 100 if not most_common.empty else None,
                    "empty_strings": (df[col] == "").sum(),
                })
        
        columns_report.append(col_info)
    
    quality_report["columns"] = columns_report
    
    return quality_report

def detect_outliers(df, method='iqr', threshold=1.5):
    """이상치를 감지합니다."""
    numeric_df = df.select_dtypes(include=['number'])
    outliers = {}
    
    if method == 'iqr':  # 사분위수 범위 기반
        for column in numeric_df.columns:
            Q1 = numeric_df[column].quantile(0.25)
            Q3 = numeric_df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outlier_indices = numeric_df[(numeric_df[column] < lower_bound) | (numeric_df[column] > upper_bound)].index
            
            outliers[column] = {
                'count': len(outlier_indices),
                'percentage': len(outlier_indices) / len(numeric_df) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'indices': outlier_indices.tolist()
            }
    elif method == 'zscore':  # Z-score 기반
        from scipy import stats
        for column in numeric_df.columns:
            z_scores = np.abs(stats.zscore(numeric_df[column].dropna()))
            outlier_indices = numeric_df.index[z_scores > threshold]
            
            outliers[column] = {
                'count': len(outlier_indices),
                'percentage': len(outlier_indices) / len(numeric_df) * 100,
                'threshold': threshold,
                'indices': outlier_indices.tolist()
            }
    
    return outliers

def handle_missing_values(df, strategy='smart'):
    """결측치를 처리합니다."""
    df_processed = df.copy()
    
    if strategy == 'smart':
        # 각 열의 특성에 따라 적절한 결측치 처리 방법 선택
        for col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count == 0:
                continue
                
            # 결측치가 80% 이상인 경우 열 제거 고려
            if missing_count / len(df) > 0.8:
                st.warning(f"'{col}' 열의 결측치가 80% 이상입니다. 이 열을 제거하는 것이 좋습니다.")
                continue
                
            # 데이터 유형에 따른 처리
            if pd.api.types.is_numeric_dtype(df[col]):
                # 수치형 데이터: 분포에 따라 평균, 중앙값 또는 0으로 대체
                if df[col].skew() > 1 or df[col].skew() < -1:  # 치우친 분포
                    df_processed[col] = df_processed[col].fillna(df_processed[col].median())
                else:  # 정규 분포에 가까움
                    df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
            
            elif pd.api.types.is_datetime64_dtype(df[col]):
                # 날짜형 데이터: 이전/이후 값으로 보간
                df_processed[col] = df_processed[col].interpolate(method='time')
            
            else:
                # 범주형/문자열 데이터: 최빈값 또는 'Unknown'으로 대체
                if df[col].nunique() < len(df) * 0.5:  # 고유값이 적은 경우
                    mode_value = df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown'
                    df_processed[col] = df_processed[col].fillna(mode_value)
                else:
                    # 고유값이 많은 경우 'Unknown'으로 대체
                    df_processed[col] = df_processed[col].fillna('Unknown')
    
    elif strategy == 'median':
        # 수치형 열만 중앙값으로 대체
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            df_processed[col] = df_processed[col].fillna(df_processed[col].median())
    
    elif strategy == 'mean':
        # 수치형 열만 평균으로 대체
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
    
    elif strategy == 'zero':
        # 수치형 열만 0으로 대체
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            df_processed[col] = df_processed[col].fillna(0)
    
    # 범주형 열은 최빈값으로 대체
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        mode_value = df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown"
        df_processed[col] = df_processed[col].fillna(mode_value)
    
    return df_processed

def scale_features(df, method='standard', columns=None):
    """특성을 스케일링합니다."""
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