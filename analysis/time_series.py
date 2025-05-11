# analysis/time_series.py - 시계열 분석 관련 함수
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm

def prepare_time_series(df, date_column, value_column):
    """시계열 데이터를 준비합니다."""
    # 날짜 열 확인
    if date_column not in df.columns:
        return None, "날짜 열이 존재하지 않습니다."
    
    # 값 열 확인
    if value_column not in df.columns:
        return None, "값 열이 존재하지 않습니다."
    
    # 날짜형으로 변환
    df = df.copy()
    df['date'] = pd.to_datetime(df[date_column])
    
    # 인덱스 설정
    ts_df = df.set_index('date')[[value_column]]
    
    # 날짜 순으로 정렬
    ts_df = ts_df.sort_index()
    
    return ts_df, None

def decompose_time_series(ts_df, value_column, period=None, model='additive'):
    """시계열 분해를 수행합니다."""
    # 주기가 지정되지 않은 경우 자동 추정
    if period is None:
        if len(ts_df) >= 365:  # 1년 이상의 데이터
            period = 365  # 일별 데이터일 경우 1년
        elif len(ts_df) >= 52:  # 1년 이상의 주별 데이터
            period = 52  # 주별 데이터
        elif len(ts_df) >= 12:  # 1년 이상의 월별 데이터
            period = 12  # 월별 데이터
        else:
            period = 4  # 기본값
    
    # 시계열 분해
    decomposition = seasonal_decompose(ts_df[value_column], model=model, period=period)
    
    # 결과를 데이터프레임으로 변환
    decomp_df = pd.DataFrame({
        'observed': decomposition.observed,
        'trend': decomposition.trend,
        'seasonal': decomposition.seasonal,
        'residual': decomposition.resid
    })
    
    return decomp_df

def calculate_rolling_statistics(ts_df, value_column, window=7):
    """이동 평균 및 표준편차를 계산합니다."""
    # 이동 평균
    rolling_mean = ts_df[value_column].rolling(window=window).mean()
    
    # 이동 표준편차
    rolling_std = ts_df[value_column].rolling(window=window).std()
    
    # 결과를 데이터프레임으로 변환
    roll_df = pd.DataFrame({
        'original': ts_df[value_column],
        f'moving_avg_{window}': rolling_mean,
        f'moving_std_{window}': rolling_std
    })
    
    return roll_df