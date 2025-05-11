# utils/data_processor.py - 데이터 전처리 관련 함수
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

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

def handle_missing_values(df, strategy='median'):
    """결측치를 처리합니다."""
    df_processed = df.copy()
    
    # 수치형 열
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    if strategy == 'median':
        for col in numeric_cols:
            df_processed[col] = df_processed[col].fillna(df_processed[col].median())
    elif strategy == 'mean':
        for col in numeric_cols:
            df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
    elif strategy == 'zero':
        for col in numeric_cols:
            df_processed[col] = df_processed[col].fillna(0)
    
    # 범주형 열은 최빈값으로 대체
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0] if not df_processed[col].mode().empty else "Unknown")
    
    return df_processed

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
            
            outliers[column] = {
                'count': len(numeric_df[(numeric_df[column] < lower_bound) | (numeric_df[column] > upper_bound)]),
                'percentage': len(numeric_df[(numeric_df[column] < lower_bound) | (numeric_df[column] > upper_bound)]) / len(numeric_df) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
    
    return outliers

def scale_features(df, method='standard'):
    """특성을 스케일링합니다."""
    numeric_df = df.select_dtypes(include=['number'])
    
    if method == 'standard':
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)
        scaled_df = pd.DataFrame(scaled_data, columns=numeric_df.columns, index=numeric_df.index)
    
    # 비수치형 열 추가
    for col in df.columns:
        if col not in numeric_df.columns:
            scaled_df[col] = df[col]
    
    return scaled_df