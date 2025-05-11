# analysis/basic_stats.py - 기초 통계 분석 관련 함수
import pandas as pd
import numpy as np

def get_basic_stats(df):
    """기초 통계량을 계산합니다."""
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

def get_categorical_stats(df):
    """범주형 변수의 통계를 계산합니다."""
    categorical_df = df.select_dtypes(include=['object', 'category', 'bool'])
    
    if categorical_df.empty:
        return pd.DataFrame()
    
    stats = {}
    
    for col in categorical_df.columns:
        value_counts = df[col].value_counts()
        unique_count = df[col].nunique()
        top_value = df[col].value_counts().index[0] if not df[col].value_counts().empty else None
        top_freq = df[col].value_counts().iloc[0] if not df[col].value_counts().empty else 0
        
        stats[col] = {
            '고유값 수': unique_count,
            '최빈값': top_value,
            '최빈값 빈도': top_freq,
            '결측치 수': df[col].isnull().sum()
        }
    
    return pd.DataFrame(stats).T

def get_data_summary(df):
    """데이터의 전반적인 요약 정보를 제공합니다."""
    summary = {
        '행 수': df.shape[0],
        '열 수': df.shape[1],
        '결측치가 있는 열 수': df.isna().any().sum(),
        '중복된 행 수': df.duplicated().sum(),
        '수치형 변수 수': len(df.select_dtypes(include=['number']).columns),
        '범주형 변수 수': len(df.select_dtypes(include=['object', 'category', 'bool']).columns),
        '날짜형 변수 수': len(df.select_dtypes(include=['datetime']).columns)
    }
    
    return summary