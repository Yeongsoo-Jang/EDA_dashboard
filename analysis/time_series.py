# analysis/time_series.py - 시계열 분석 관련 클래스
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Optional, Tuple, Union, Any
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import time
from datetime import datetime, timedelta
import pytz

# DataAnalyzer 클래스 임포트
from analysis.basic_stats import DataAnalyzer


class TimeSeriesAnalyzer(DataAnalyzer):
    """시계열 데이터 분석을 위한 기본 클래스"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Parameters:
        -----------
        df : DataFrame
            분석할 데이터프레임
        """
        super().__init__(df)
        self.ts_df = None
        self.date_column = None
        self.value_column = None
        self.frequency = None
        self.decomposition = None
        self.stationarity_tests = {}
        self.forecast_results = {}
        
        # 클래스 초기화 시 날짜 열 자동 감지
        self._detect_date_column()
    
    def _detect_date_column(self) -> None:
        """데이터프레임에서 날짜 열을 자동으로 감지합니다."""
        # 이미 datetime 유형인 열 확인
        date_cols = self.df.select_dtypes(include=['datetime64']).columns.tolist()
        
        if date_cols:
            self.date_column = date_cols[0]  # 첫 번째 날짜 열 선택
        else:
            # 열 이름에 날짜 관련 키워드가 있는지 확인
            date_keywords = ['date', 'time', 'year', 'month', 'day', '일자', '날짜', '연도', '월', '일']
            for col in self.df.columns:
                if any(keyword in col.lower() for keyword in date_keywords):
                    # 자동 변환 시도
                    try:
                        pd.to_datetime(self.df[col])
                        self.date_column = col
                        break
                    except:
                        pass
    
    def prepare_time_series(self, date_column: str = None, value_column: str = None, 
                          resample_freq: str = None, agg_func: str = 'mean',
                          fill_method: str = 'ffill') -> Tuple[pd.DataFrame, str]:
        """시계열 데이터를 준비합니다.
        
        Parameters:
        -----------
        date_column : str, default=None
            날짜 열 (None이면 자동 감지)
        value_column : str, default=None
            값 열 (None이면 첫 번째 수치형 열)
        resample_freq : str, default=None
            리샘플링 주기 ('D', 'W', 'M', 'Q', 'Y' 등, None이면 리샘플링 안 함)
        agg_func : str, default='mean'
            리샘플링 시 집계 함수 ('mean', 'sum', 'min', 'max')
        fill_method : str, default='ffill'
            결측치 처리 방법 ('ffill', 'bfill', 'interpolate', None)
        
        Returns:
        --------
        tuple
            (시계열 데이터프레임, 오류 메시지)
        """
        # 날짜 열 확인
        if date_column is not None:
            self.date_column = date_column
        elif self.date_column is None:
            # 자동 감지 시도
            self._detect_date_column()
            
            if self.date_column is None:
                return None, "날짜 열을 찾을 수 없습니다. date_column 매개변수로 날짜 열을 지정하세요."
        
        # 값 열 확인
        if value_column is not None:
            self.value_column = value_column
        else:
            # 수치형 열 중 첫 번째 열 선택
            numeric_cols = [col for col in self.numeric_cols if col != self.date_column]
            if numeric_cols:
                self.value_column = numeric_cols[0]
            else:
                return None, "분석할 수치형 열이 없습니다. value_column 매개변수로 값 열을 지정하세요."
        
        # 날짜 열이 datetime 유형이 아니면 변환
        if not pd.api.types.is_datetime64_dtype(self.df[self.date_column]):
            try:
                self.df['date'] = pd.to_datetime(self.df[self.date_column])
                date_col = 'date'
            except:
                return None, f"'{self.date_column}' 열을 날짜 형식으로 변환할 수 없습니다."
        else:
            date_col = self.date_column
        
        # 시계열 데이터프레임 생성
        self.ts_df = self.df[[date_col, self.value_column]].copy()
        
        # 결측치 제거
        if fill_method:
            if fill_method == 'interpolate':
                self.ts_df[self.value_column] = self.ts_df[self.value_column].interpolate(method='time')
            else:
                self.ts_df[self.value_column] = self.ts_df[self.value_column].fillna(method=fill_method)
        
        # 날짜 순으로 정렬
        self.ts_df = self.ts_df.sort_values(date_col)
        
        # 중복 날짜 처리
        if self.ts_df[date_col].duplicated().any():
            # 중복된 날짜는 집계 함수로 처리
            if agg_func == 'mean':
                self.ts_df = self.ts_df.groupby(date_col)[self.value_column].mean().reset_index()
            elif agg_func == 'sum':
                self.ts_df = self.ts_df.groupby(date_col)[self.value_column].sum().reset_index()
            elif agg_func == 'min':
                self.ts_df = self.ts_df.groupby(date_col)[self.value_column].min().reset_index()
            elif agg_func == 'max':
                self.ts_df = self.ts_df.groupby(date_col)[self.value_column].max().reset_index()
        
        # 인덱스 설정
        self.ts_df = self.ts_df.set_index(date_col)
        
        # 리샘플링 (필요한 경우)
        if resample_freq:
            try:
                if agg_func == 'mean':
                    self.ts_df = self.ts_df.resample(resample_freq).mean()
                elif agg_func == 'sum':
                    self.ts_df = self.ts_df.resample(resample_freq).sum()
                elif agg_func == 'min':
                    self.ts_df = self.ts_df.resample(resample_freq).min()
                elif agg_func == 'max':
                    self.ts_df = self.ts_df.resample(resample_freq).max()
                
                self.frequency = resample_freq
            except Exception as e:
                return self.ts_df, f"리샘플링 중 오류 발생: {str(e)}"
        
        # 자연 주기 자동 감지
        if self.frequency is None:
            self._detect_frequency()
        
        return self.ts_df, None
    
    def _detect_frequency(self) -> None:
        """시계열 데이터의 주기를 자동으로 감지합니다."""
        if self.ts_df is None:
            return
        
        # 시간 간격 계산
        time_diffs = self.ts_df.index.to_series().diff().dropna()
        
        if time_diffs.empty:
            return
        
        # 가장 빈번한 시간 간격 확인
        most_common_diff = time_diffs.mode().iloc[0]
        
        # 주기 결정
        days = most_common_diff.days
        seconds = most_common_diff.seconds
        
        if days == 0 and seconds < 60:
            self.frequency = 'S'  # 초
        elif days == 0 and seconds < 3600:
            self.frequency = 'T'  # 분
        elif days == 0 and seconds < 86400:
            self.frequency = 'H'  # 시간
        elif days == 1:
            self.frequency = 'D'  # 일
        elif 6 <= days <= 8:
            self.frequency = 'W'  # 주
        elif 28 <= days <= 31:
            self.frequency = 'M'  # 월
        elif 89 <= days <= 93:
            self.frequency = 'Q'  # 분기
        elif 364 <= days <= 366:
            self.frequency = 'Y'  # 년
        else:
            # 기본값
            self.frequency = 'D'
    
    @st.cache_data(ttl=3600)
    def decompose_time_series(self, model: str = 'additive', period: int = None) -> Dict[str, pd.Series]:
        """시계열 분해를 수행합니다.
        
        Parameters:
        -----------
        model : str, default='additive'
            분해 모델 ('additive' 또는 'multiplicative')
        period : int, default=None
            주기 (None이면 자동 감지)
        
        Returns:
        --------
        dict
            분해 결과 구성 요소 (trend, seasonal, resid)
        """
        if self.ts_df is None:
            raise ValueError("먼저 prepare_time_series 메서드를 호출하세요.")
        
        # 주기가 지정되지 않은 경우 자동 추정
        if period is None:
            if self.frequency == 'S':
                period = 60  # 초 (1분)
            elif self.frequency == 'T':
                period = 60  # 분 (1시간)
            elif self.frequency == 'H':
                period = 24  # 시간 (1일)
            elif self.frequency == 'D':
                period = 7  # 일 (1주)
            elif self.frequency == 'W':
                period = 52  # 주 (1년)
            elif self.frequency == 'M':
                period = 12  # 월 (1년)
            elif self.frequency == 'Q':
                period = 4  # 분기 (1년)
            else:
                # 주기를 추정하기 위한 휴리스틱
                if len(self.ts_df) >= 365:
                    period = 365  # 일별 데이터라고 가정
                elif len(self.ts_df) >= 52:
                    period = 52  # 주별 데이터라고 가정
                elif len(self.ts_df) >= 12:
                    period = 12  # 월별 데이터라고 가정
                else:
                    period = min(len(self.ts_df) // 2, 4)  # 기본값
        
        # 시계열 분해
        try:
            if model not in ['additive', 'multiplicative']:
                model = 'additive'  # 기본값
                
            self.decomposition = seasonal_decompose(
                self.ts_df[self.value_column],
                model=model,
                period=period
            )
            
            # 결과를 데이터프레임으로 변환
            decomp_df = pd.DataFrame({
                'observed': self.decomposition.observed,
                'trend': self.decomposition.trend,
                'seasonal': self.decomposition.seasonal,
                'residual': self.decomposition.resid
            })
            
            return decomp_df
            
        except Exception as e:
            print(f"시계열 분해 중 오류 발생: {str(e)}")
            # 최소한의 결과 반환
            return pd.DataFrame({
                'observed': self.ts_df[self.value_column],
                'error': str(e)
            })
    
    def calculate_rolling_statistics(self, window: int = 7, center: bool = False) -> pd.DataFrame:
        """이동 통계량(평균, 표준편차 등)을 계산합니다.
        
        Parameters:
        -----------
        window : int, default=7
            이동 평균 창 크기
        center : bool, default=False
            중앙값 기준 계산 여부
        
        Returns:
        --------
        DataFrame
            이동 통계량 데이터프레임
        """
        if self.ts_df is None:
            raise ValueError("먼저 prepare_time_series 메서드를 호출하세요.")
        
        # 자동 창 크기 조정
        if window > len(self.ts_df) // 2:
            window = max(len(self.ts_df) // 10, 2)
            print(f"창 크기가 너무 큽니다. {window}로 조정합니다.")
        
        # 이동 통계량 계산
        roll_df = pd.DataFrame({
            'original': self.ts_df[self.value_column],
            f'rolling_mean_{window}': self.ts_df[self.value_column].rolling(window=window, center=center).mean(),
            f'rolling_std_{window}': self.ts_df[self.value_column].rolling(window=window, center=center).std(),
            f'rolling_min_{window}': self.ts_df[self.value_column].rolling(window=window, center=center).min(),
            f'rolling_max_{window}': self.ts_df[self.value_column].rolling(window=window, center=center).max()
        })
        
        # 지수 이동 평균 (EMA)
        roll_df[f'exp_mean_{window}'] = self.ts_df[self.value_column].ewm(span=window).mean()
        
        return roll_df
    
    def test_stationarity(self) -> Dict[str, Any]:
        """시계열 데이터의 정상성을 검정합니다.
        
        Returns:
        --------
        dict
            정상성 검정 결과
        """
        if self.ts_df is None:
            raise ValueError("먼저 prepare_time_series 메서드를 호출하세요.")
        
        # 결측치 제거
        series = self.ts_df[self.value_column].dropna()
        
        results = {}
        
        # ADF 검정 (귀무가설: 단위근 존재 = 비정상)
        try:
            adf_result = adfuller(series)
            results['adf'] = {
                'test_statistic': adf_result[0],
                'p_value': adf_result[1],
                'critical_values': adf_result[4],
                'is_stationary': adf_result[1] < 0.05  # p-value < 0.05이면 정상
            }
        except:
            results['adf'] = {'error': '충분한 데이터가 없거나 검정할 수 없습니다.'}
        
        # KPSS 검정 (귀무가설: 정상성 = 정상)
        try:
            kpss_result = kpss(series)
            results['kpss'] = {
                'test_statistic': kpss_result[0],
                'p_value': kpss_result[1],
                'critical_values': kpss_result[3],
                'is_stationary': kpss_result[1] > 0.05  # p-value > 0.05이면 정상
            }
        except:
            results['kpss'] = {'error': '충분한 데이터가 없거나 검정할 수 없습니다.'}
        
        # 두 검정 결과 종합
        if ('is_stationary' in results['adf'] and 'is_stationary' in results['kpss']):
            if results['adf']['is_stationary'] and results['kpss']['is_stationary']:
                results['conclusion'] = '시계열이 정상입니다 (ADF, KPSS 모두 정상)'
            elif results['adf']['is_stationary'] and not results['kpss']['is_stationary']:
                results['conclusion'] = '추세 정상성 (ADF에서는 정상, KPSS에서는 비정상)'
            elif not results['adf']['is_stationary'] and results['kpss']['is_stationary']:
                results['conclusion'] = '차분이 필요할 수 있습니다 (ADF에서는 비정상, KPSS에서는 정상)'
            else:
                results['conclusion'] = '시계열이 비정상입니다 (ADF, KPSS 모두 비정상)'
        
        self.stationarity_tests = results
        return results
    
    def make_stationary(self, method: str = 'diff', order: int = 1) -> pd.Series:
        """시계열 데이터를 정상화합니다.
        
        Parameters:
        -----------
        method : str, default='diff'
            정상화 방법 ('diff', 'log', 'log_diff', 'pct_change')
        order : int, default=1
            차분 차수
        
        Returns:
        --------
        Series
            정상화된 시계열
        """
        if self.ts_df is None:
            raise ValueError("먼저 prepare_time_series 메서드를 호출하세요.")
        
        series = self.ts_df[self.value_column]
        
        if method == 'diff':
            # 차분
            return series.diff(order).dropna()
        elif method == 'log':
            # 로그 변환 (음수 값이 있으면 조정)
            if series.min() <= 0:
                min_val = series.min()
                adjusted = series - min_val + 1
                return np.log(adjusted)
            return np.log(series)
        elif method == 'log_diff':
            # 로그 변환 후 차분
            if series.min() <= 0:
                min_val = series.min()
                adjusted = series - min_val + 1
                log_series = np.log(adjusted)
            else:
                log_series = np.log(series)
            return log_series.diff(order).dropna()
        elif method == 'pct_change':
            # 백분율 변화
            return series.pct_change(periods=order).dropna()
        else:
            raise ValueError(f"지원하지 않는 정상화 방법: {method}")
    
    def plot_acf_pacf(self, lags: int = 40, method: str = None,
                    alpha: float = 0.05) -> plt.Figure:
        """자기상관함수(ACF)와 편자기상관함수(PACF) 플롯을 생성합니다.
        
        Parameters:
        -----------
        lags : int, default=40
            지연 수
        method : str, default=None
            정상화 방법 (None은 원본 시계열 사용)
        alpha : float, default=0.05
            신뢰구간 수준
        
        Returns:
        --------
        matplotlib.figure.Figure
            ACF, PACF 플롯
        """
        if self.ts_df is None:
            raise ValueError("먼저 prepare_time_series 메서드를 호출하세요.")
        
        # 최대 지연 조정
        max_lags = min(lags, len(self.ts_df) // 2)
        
        # 시계열 선택
        if method:
            series = self.make_stationary(method=method)
        else:
            series = self.ts_df[self.value_column].dropna()
        
        # 그래프 생성
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # ACF 플롯
        plot_acf(series, lags=max_lags, alpha=alpha, ax=ax1)
        ax1.set_title('자기상관함수(ACF)')
        
        # PACF 플롯
        plot_pacf(series, lags=max_lags, alpha=alpha, ax=ax2)
        ax2.set_title('편자기상관함수(PACF)')
        
        plt.tight_layout()
        return fig
    
    def get_seasonality_info(self) -> Dict[str, Any]:
        """시계열의 계절성 정보를 분석합니다.
        
        Returns:
        --------
        dict
            계절성 분석 결과
        """
        if self.ts_df is None:
            raise ValueError("먼저 prepare_time_series 메서드를 호출하세요.")
        
        # 결측치 제거
        series = self.ts_df[self.value_column].dropna()
        
        results = {}
        
        # 주기성 휴리스틱 검출
        acf_values = acf(series, nlags=min(len(series) // 2, 100))
        
        # ACF 값에서 피크 찾기
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(acf_values, distance=3)
        
        if len(peaks) > 0:
            results['acf_peaks'] = peaks.tolist()
            results['suggested_seasonal_periods'] = peaks[0] if peaks[0] > 1 else None
        else:
            results['acf_peaks'] = []
            results['suggested_seasonal_periods'] = None
        
        # 일/주/월/분기/연 계절성 확인
        dt_index = self.ts_df.index
        
        # 날짜 속성을 사용하여 계절성 검사
        day_of_week_means = series.groupby(dt_index.dayofweek).mean()
        month_means = series.groupby(dt_index.month).mean()
        
        # 변동 계수 (CV)를 사용하여 계절성 강도 측정 (표준편차/평균)
        day_of_week_cv = day_of_week_means.std() / day_of_week_means.mean() if day_of_week_means.mean() != 0 else 0
        month_cv = month_means.std() / month_means.mean() if month_means.mean() != 0 else 0
        
        results['day_of_week_seasonality'] = {
            'strength': day_of_week_cv,
            'means': day_of_week_means.to_dict(),
            'significant': day_of_week_cv > 0.1  # 임의의 임계값
        }
        
        results['monthly_seasonality'] = {
            'strength': month_cv,
            'means': month_means.to_dict(),
            'significant': month_cv > 0.1  # 임의의 임계값
        }
        
        # 정상성이 이미 테스트되었으면 추가
        if self.stationarity_tests:
            results['stationarity'] = self.stationarity_tests
        
        return results


class TimeSeriesForecaster(TimeSeriesAnalyzer):
    """시계열 예측을 위한 클래스"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Parameters:
        -----------
        df : DataFrame
            분석할 데이터프레임
        """
        super().__init__(df)
        self.train_data = None
        self.test_data = None
        self.forecasts = {}
        self.model = None
        self.forecast_horizon = None
        self.metrics = {}
    
    def split_data(self, test_size: float = 0.2, n_test_periods: int = None) -> Tuple[pd.Series, pd.Series]:
        """데이터를 훈련 세트와 테스트 세트로 분할합니다.
        
        Parameters:
        -----------
        test_size : float, default=0.2
            테스트 세트 비율
        n_test_periods : int, default=None
            테스트 세트 기간 수 (지정하면 test_size 무시)
        
        Returns:
        --------
        tuple
            (훈련 데이터, 테스트 데이터)
        """
        if self.ts_df is None:
            raise ValueError("먼저 prepare_time_series 메서드를 호출하세요.")
        
        series = self.ts_df[self.value_column].dropna()
        
        if n_test_periods:
            # 테스트 세트 기간 수 기준 분할
            split_idx = len(series) - n_test_periods
        else:
            # 비율 기준 분할
            split_idx = int(len(series) * (1 - test_size))
        
        split_idx = max(1, min(split_idx, len(series) - 1))  # 범위 제한
        
        self.train_data = series.iloc[:split_idx].copy()
        self.test_data = series.iloc[split_idx:].copy()
        
        return self.train_data, self.test_data
    
    def fit_naive_forecast(self, method: str = 'last', seasonal_period: int = None) -> Dict[str, Any]:
        """나이브 예측 모델을 적용합니다.
        
        Parameters:
        -----------
        method : str, default='last'
            예측 방법 ('last', 'mean', 'drift', 'seasonal')
        seasonal_period : int, default=None
            계절성 주기 (seasonal 방법에 필요)
        
        Returns:
        --------
        dict
            예측 결과
        """
        if self.train_data is None or self.test_data is None:
            self.split_data()
        
        train = self.train_data
        test = self.test_data
        
        # 예측 기간
        self.forecast_horizon = len(test)
        
        # 예측 수행
        if method == 'last':
            # 마지막 값으로 예측
            forecast = pd.Series(
                [train.iloc[-1]] * self.forecast_horizon,
                index=test.index
            )
            model_name = 'Naive (Last Value)'
        
        elif method == 'mean':
            # 평균값으로 예측
            forecast = pd.Series(
                [train.mean()] * self.forecast_horizon,
                index=test.index
            )
            model_name = 'Naive (Mean)'
        
        elif method == 'drift':
            # 추세를 고려한 예측
            last_value = train.iloc[-1]
            slope = (train.iloc[-1] - train.iloc[0]) / (len(train) - 1)
            
            forecast = pd.Series(
                [last_value + slope * (i + 1) for i in range(self.forecast_horizon)],
                index=test.index
            )
            model_name = 'Naive (Drift)'
        
        elif method == 'seasonal':
            # 계절성을 고려한 예측
            if seasonal_period is None:
                # 주기 자동 감지
                if self.frequency == 'D':
                    seasonal_period = 7  # 일별 데이터의 주간 계절성
                elif self.frequency == 'M':
                    seasonal_period = 12  # 월별 데이터의 연간 계절성
                elif self.frequency == 'Q':
                    seasonal_period = 4  # 분기별 데이터의 연간 계절성
                else:
                    seasonal_period = min(len(train) // 2, 12)  # 기본값
            
            # 계절성 인덱스
            forecast = pd.Series(
                [train.iloc[-seasonal_period + (i % seasonal_period)] for i in range(self.forecast_horizon)],
                index=test.index
            )
            model_name = f'Seasonal Naive (Period={seasonal_period})'
        
        else:
            raise ValueError(f"지원하지 않는 나이브 예측 방법: {method}")
        
        # 예측 결과 저장
        self.forecasts[model_name] = {
            'forecast': forecast,
            'actual': test,
            'train': train,
            'mse': mean_squared_error(test, forecast),
            'mae': mean_absolute_error(test, forecast),
            'rmse': np.sqrt(mean_squared_error(test, forecast)),
            'mape': np.mean(np.abs((test - forecast) / test)) * 100 if (test != 0).all() else np.nan
        }
        
        return self.forecasts[model_name]
    
    def fit_exponential_smoothing(self, trend: str = None, seasonal: str = None,
                                seasonal_periods: int = None, damped: bool = False,
                                use_boxcox: bool = False) -> Dict[str, Any]:
        """지수 평활(Exponential Smoothing) 모델을 적합시킵니다.
        
        Parameters:
        -----------
        trend : str, default=None
            추세 유형 (None, 'add', 'mul')
        seasonal : str, default=None
            계절성 유형 (None, 'add', 'mul')
        seasonal_periods : int, default=None
            계절성 주기
        damped : bool, default=False
            감쇠 추세 사용 여부
        use_boxcox : bool, default=False
            Box-Cox 변환 사용 여부
        
        Returns:
        --------
        dict
            예측 결과
        """
        if self.train_data is None or self.test_data is None:
            self.split_data()
        
        train = self.train_data
        test = self.test_data
        
        # 계절성 주기 자동 설정
        if seasonal and seasonal_periods is None:
            # 주기 자동 감지
            if self.frequency == 'D':
                seasonal_periods = 7  # 일별 데이터의 주간 계절성
            elif self.frequency == 'M':
                seasonal_periods = 12  # 월별 데이터의 연간 계절성
            elif self.frequency == 'Q':
                seasonal_periods = 4  # 분기별 데이터의 연간 계절성
            else:
                seasonal_periods = min(len(train) // 2, 12)  # 기본값
        
        # 예측 기간
        self.forecast_horizon = len(test)
        
        model_name = f'ETS'
        components = []
        
        if trend:
            components.append(f"trend={trend[0].upper()}")
            if damped:
                components.append("damped")
        
        if seasonal:
            components.append(f"seasonal={seasonal[0].upper()}")
            components.append(f"period={seasonal_periods}")
        
        if components:
            model_name += f" ({', '.join(components)})"
        
        try:
            # 간단한 지수 평활(SES) 모델 (추세, 계절성 없음)
            if trend is None and seasonal is None:
                model = SimpleExpSmoothing(train)
                fit_model = model.fit()
                
            # 완전한 지수 평활 모델
            else:
                model = ExponentialSmoothing(
                    train,
                    trend=trend,
                    seasonal=seasonal,
                    seasonal_periods=seasonal_periods,
                    damped_trend=damped,
                    use_boxcox=use_boxcox
                )
                fit_model = model.fit()
            
            # 예측 생성
            forecast = fit_model.forecast(self.forecast_horizon)
            
            # 인덱스 조정
            forecast.index = test.index
            
            # 평가 지표 계산
            mse = mean_squared_error(test, forecast)
            mae = mean_absolute_error(test, forecast)
            rmse = np.sqrt(mse)
            
            # MAPE 계산 (0 값 주의)
            if (test != 0).all():
                mape = np.mean(np.abs((test - forecast) / test)) * 100
            else:
                mape = np.nan
            
            # 결과 저장
            self.forecasts[model_name] = {
                'forecast': forecast,
                'actual': test,
                'train': train,
                'model': fit_model,
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'aic': fit_model.aic,
                'bic': fit_model.bic
            }
            
            return self.forecasts[model_name]
        
        except Exception as e:
            print(f"지수 평활 모델 적합 중 오류 발생: {str(e)}")
            return {
                'error': str(e),
                'forecast': None,
                'actual': test,
                'train': train
            }
    
    def fit_arima(self, order: Tuple[int, int, int] = None, seasonal_order: Tuple[int, int, int, int] = None,
                seasonal_period: int = None, auto: bool = False) -> Dict[str, Any]:
        """ARIMA 또는 SARIMA 모델을 적합시킵니다.
        
        Parameters:
        -----------
        order : tuple, default=None
            ARIMA 차수 (p, d, q) (auto=True이면 무시)
        seasonal_order : tuple, default=None
            계절성 ARIMA 차수 (P, D, Q, s) (auto=True이면 무시)
        seasonal_period : int, default=None
            계절성 주기 (seasonal_order가 지정된 경우 필요)
        auto : bool, default=False
            자동 ARIMA 사용 여부
        
        Returns:
        --------
        dict
            예측 결과
        """
        if self.train_data is None or self.test_data is None:
            self.split_data()
        
        train = self.train_data
        test = self.test_data
        
        # 예측 기간
        self.forecast_horizon = len(test)
        
        try:
            # 자동 ARIMA
            if auto:
                try:
                    from pmdarima import auto_arima
                    
                    # 자동 ARIMA 수행
                    auto_model = auto_arima(
                        train,
                        seasonal=True,
                        suppress_warnings=True,
                        error_action='ignore',
                        stepwise=True
                    )
                    
                    # 최적 파라미터 추출
                    order = auto_model.order
                    seasonal_order = auto_model.seasonal_order
                    
                    model_name = f'Auto ARIMA {order}'
                    if seasonal_order[0] + seasonal_order[1] + seasonal_order[2] > 0:
                        model_name += f' {seasonal_order}'
                
                except ImportError:
                    print("자동 ARIMA 사용을 위해 pmdarima 패키지가 필요합니다. pip install pmdarima")
                    auto = False
            
            # 수동 ARIMA
            if not auto:
                if order is None:
                    order = (1, 1, 1)  # 기본값
                
                if seasonal_order is not None and seasonal_period is None:
                    # 주기 자동 감지
                    if self.frequency == 'D':
                        seasonal_period = 7  # 일별 데이터의 주간 계절성
                    elif self.frequency == 'M':
                        seasonal_period = 12  # 월별 데이터의 연간 계절성
                    elif self.frequency == 'Q':
                        seasonal_period = 4  # 분기별 데이터의 연간 계절성
                    else:
                        seasonal_period = min(len(train) // 2, 12)  # 기본값
                    
                    seasonal_order = (*seasonal_order[:3], seasonal_period)
                
                model_name = f'ARIMA {order}'
                if seasonal_order:
                    model_name = f'SARIMA {order} {seasonal_order}'
            
            # 모델 적합
            if seasonal_order and seasonal_order[0] + seasonal_order[1] + seasonal_order[2] > 0:
                # SARIMA 모델
                model = SARIMAX(
                    train,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
            else:
                # ARIMA 모델
                model = ARIMA(
                    train,
                    order=order
                )
            
            fit_model = model.fit()
            
            # 예측 생성
            forecast = fit_model.forecast(self.forecast_horizon)
            
            # 인덱스 조정
            forecast.index = test.index
            
            # 평가 지표 계산
            mse = mean_squared_error(test, forecast)
            mae = mean_absolute_error(test, forecast)
            rmse = np.sqrt(mse)
            
            # MAPE 계산 (0 값 주의)
            if (test != 0).all():
                mape = np.mean(np.abs((test - forecast) / test)) * 100
            else:
                mape = np.nan
            
            # 결과 저장
            self.forecasts[model_name] = {
                'forecast': forecast,
                'actual': test,
                'train': train,
                'model': fit_model,
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'aic': fit_model.aic,
                'bic': fit_model.bic if hasattr(fit_model, 'bic') else None
            }
            
            return self.forecasts[model_name]
        
        except Exception as e:
            print(f"ARIMA 모델 적합 중 오류 발생: {str(e)}")
            return {
                'error': str(e),
                'forecast': None,
                'actual': test,
                'train': train
            }
    
    def compare_models(self) -> pd.DataFrame:
        """적합시킨 모든 모델의 성능을 비교합니다.
        
        Returns:
        --------
        DataFrame
            모델 비교 결과 데이터프레임
        """
        if not self.forecasts:
            raise ValueError("먼저 예측 모델을 적합시키세요.")
        
        # 결과 추출
        results = []
        
        for model_name, forecast_dict in self.forecasts.items():
            if 'error' in forecast_dict:
                # 오류가 있는 모델은 제외
                continue
            
            result = {
                'model': model_name,
                'mse': forecast_dict['mse'],
                'rmse': forecast_dict['rmse'],
                'mae': forecast_dict['mae'],
                'mape': forecast_dict['mape']
            }
            
            # AIC, BIC 추가 (있는 경우)
            if 'aic' in forecast_dict:
                result['aic'] = forecast_dict['aic']
            if 'bic' in forecast_dict and forecast_dict['bic'] is not None:
                result['bic'] = forecast_dict['bic']
            
            results.append(result)
        
        # 데이터프레임 변환 및 정렬
        if results:
            results_df = pd.DataFrame(results)
            # RMSE 기준 정렬
            results_df = results_df.sort_values('rmse')
            
            # 최고 모델 표시
            best_model = results_df.iloc[0]['model']
            print(f"최고 성능 모델 (RMSE 기준): {best_model}")
            
            return results_df
        else:
            return pd.DataFrame(columns=['model', 'mse', 'rmse', 'mae', 'mape', 'aic', 'bic'])
    
    def forecast_future(self, steps: int = 10, model_name: str = None,
                      interval: bool = True, alpha: float = 0.05) -> Dict[str, Any]:
        """미래를 예측합니다.
        
        Parameters:
        -----------
        steps : int, default=10
            예측 기간 수
        model_name : str, default=None
            사용할 모델 이름 (None이면 최고 성능 모델 사용)
        interval : bool, default=True
            예측 구간 계산 여부
        alpha : float, default=0.05
            예측 구간 신뢰수준
        
        Returns:
        --------
        dict
            미래 예측 결과
        """
        if not self.forecasts:
            raise ValueError("먼저 예측 모델을 적합시키세요.")
        
        # 모델 선택
        if model_name is None:
            # 최고 성능 모델 선택 (RMSE 기준)
            compare_df = self.compare_models()
            if not compare_df.empty:
                model_name = compare_df.iloc[0]['model']
            else:
                raise ValueError("유효한 모델이 없습니다.")
        
        if model_name not in self.forecasts:
            raise ValueError(f"'{model_name}' 모델이 없습니다.")
        
        forecast_dict = self.forecasts[model_name]
        
        # 오류 확인
        if 'error' in forecast_dict or 'model' not in forecast_dict:
            raise ValueError(f"'{model_name}' 모델이 유효하지 않습니다.")
        
        model = forecast_dict['model']
        
        # 미래 날짜 생성
        last_date = self.ts_df.index[-1]
        
        if self.frequency == 'D':
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps, freq='D')
        elif self.frequency == 'W':
            future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=steps, freq='W')
        elif self.frequency == 'M':
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=31), periods=steps, freq='MS')
        elif self.frequency == 'Q':
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=92), periods=steps, freq='QS')
        else:
            # 기본값은 일별
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps, freq='D')
        
        try:
            # 모델 유형에 따른 예측
            if 'Naive' in model_name:
                # 나이브 예측 모델
                if 'Last Value' in model_name:
                    forecast = pd.Series([self.ts_df[self.value_column].iloc[-1]] * steps, index=future_dates)
                    forecast_lower = None
                    forecast_upper = None
                
                elif 'Mean' in model_name:
                    forecast = pd.Series([self.ts_df[self.value_column].mean()] * steps, index=future_dates)
                    forecast_lower = None
                    forecast_upper = None
                
                elif 'Drift' in model_name:
                    series = self.ts_df[self.value_column]
                    last_value = series.iloc[-1]
                    slope = (series.iloc[-1] - series.iloc[0]) / (len(series) - 1)
                    
                    forecast = pd.Series(
                        [last_value + slope * (i + 1) for i in range(steps)],
                        index=future_dates
                    )
                    forecast_lower = None
                    forecast_upper = None
                
                elif 'Seasonal' in model_name:
                    import re
                    # 정규식으로 주기 추출
                    match = re.search(r'Period=(\d+)', model_name)
                    seasonal_period = int(match.group(1)) if match else 7  # 기본값
                    
                    series = self.ts_df[self.value_column]
                    forecast = pd.Series(
                        [series.iloc[-seasonal_period + (i % seasonal_period)] for i in range(steps)],
                        index=future_dates
                    )
                    forecast_lower = None
                    forecast_upper = None
                
                else:
                    raise ValueError(f"지원하지 않는 나이브 예측 모델: {model_name}")
            
            elif 'ETS' in model_name:
                # 지수 평활 모델
                # forecast = model.forecast(steps)
                forecast_results = model.get_forecast(steps)
                forecast = forecast_results.predicted_mean
                forecast.index = future_dates
                
                if interval:
                    conf_int = forecast_results.conf_int(alpha=alpha)
                    forecast_lower = conf_int.iloc[:, 0]
                    forecast_upper = conf_int.iloc[:, 1]
                    forecast_lower.index = future_dates
                    forecast_upper.index = future_dates
                else:
                    forecast_lower = None
                    forecast_upper = None
            
            elif 'ARIMA' in model_name or 'SARIMA' in model_name:
                # ARIMA 또는 SARIMA 모델
                forecast_results = model.get_forecast(steps)
                forecast = forecast_results.predicted_mean
                forecast.index = future_dates
                
                if interval:
                    conf_int = forecast_results.conf_int(alpha=alpha)
                    forecast_lower = conf_int.iloc[:, 0]
                    forecast_upper = conf_int.iloc[:, 1]
                    forecast_lower.index = future_dates
                    forecast_upper.index = future_dates
                else:
                    forecast_lower = None
                    forecast_upper = None
            
            else:
                raise ValueError(f"지원하지 않는 예측 모델: {model_name}")
            
            # 결과 저장
            future_forecast = {
                'forecast': forecast,
                'lower_bound': forecast_lower,
                'upper_bound': forecast_upper,
                'model': model_name,
                'start_date': future_dates[0],
                'end_date': future_dates[-1]
            }
            
            return future_forecast
        
        except Exception as e:
            print(f"미래 예측 중 오류 발생: {str(e)}")
            return {
                'error': str(e),
                'model': model_name
            }
    
    def plot_forecast_results(self, model_name: str = None, future_steps: int = 0,
                           include_train: bool = True, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """예측 결과를 시각화합니다.
        
        Parameters:
        -----------
        model_name : str, default=None
            표시할 모델 이름 (None이면 모든 모델 표시)
        future_steps : int, default=0
            표시할 미래 예측 기간 수
        include_train : bool, default=True
            훈련 데이터 포함 여부
        figsize : tuple, default=(12, 6)
            그래프 크기
        
        Returns:
        --------
        matplotlib.figure.Figure
            예측 결과 그래프
        """
        if not self.forecasts:
            raise ValueError("먼저 예측 모델을 적합시키세요.")
        
        # 모델 선택
        if model_name is None:
            # 최고 성능 모델 선택 (RMSE 기준)
            compare_df = self.compare_models()
            if not compare_df.empty:
                model_name = compare_df.iloc[0]['model']
            else:
                # 첫 번째 모델 선택
                model_name = list(self.forecasts.keys())[0]
        
        if model_name not in self.forecasts:
            raise ValueError(f"'{model_name}' 모델이 없습니다.")
        
        forecast_dict = self.forecasts[model_name]
        
        # 오류 확인
        if 'error' in forecast_dict:
            raise ValueError(f"'{model_name}' 모델이 유효하지 않습니다.")
        
        # 그래프 생성
        fig, ax = plt.subplots(figsize=figsize)
        
        # 훈련 데이터 표시
        if include_train and 'train' in forecast_dict:
            ax.plot(forecast_dict['train'].index, forecast_dict['train'], 
                   label='훈련 데이터', color='blue')
        
        # 테스트 데이터 표시
        if 'actual' in forecast_dict:
            ax.plot(forecast_dict['actual'].index, forecast_dict['actual'], 
                   label='실제 데이터', color='green')
        
        # 예측 표시
        if 'forecast' in forecast_dict:
            ax.plot(forecast_dict['forecast'].index, forecast_dict['forecast'], 
                   label='예측', color='red', linestyle='--')
        
        # 미래 예측 표시
        if future_steps > 0:
            try:
                future_forecast = self.forecast_future(steps=future_steps, model_name=model_name)
                
                if 'forecast' in future_forecast:
                    ax.plot(future_forecast['forecast'].index, future_forecast['forecast'], 
                           label='미래 예측', color='purple', linestyle='-.')
                    
                    # 예측 구간 표시
                    if 'lower_bound' in future_forecast and future_forecast['lower_bound'] is not None:
                        ax.fill_between(
                            future_forecast['forecast'].index,
                            future_forecast['lower_bound'],
                            future_forecast['upper_bound'],
                            color='lavender', alpha=0.5,
                            label='95% 예측 구간'
                        )
            except Exception as e:
                print(f"미래 예측 표시 중 오류 발생: {str(e)}")
        
        # 그래프 설정
        ax.set_title(f'{model_name} 모델 예측 결과')
        ax.set_xlabel('날짜')
        ax.set_ylabel(self.value_column)
        ax.legend()
        ax.grid(True)
        
        # 성능 지표 표시
        if all(k in forecast_dict for k in ['mse', 'rmse', 'mae']):
            mse = forecast_dict['mse']
            rmse = forecast_dict['rmse']
            mae = forecast_dict['mae']
            
            metrics_text = f'RMSE: {rmse:.4f}, MAE: {mae:.4f}'
            
            # MAPE 추가 (있는 경우)
            if 'mape' in forecast_dict and not np.isnan(forecast_dict['mape']):
                mape = forecast_dict['mape']
                metrics_text += f', MAPE: {mape:.2f}%'
            
            ax.text(0.05, 0.05, metrics_text, transform=ax.transAxes, 
                   bbox=dict(facecolor='white', alpha=0.7))
        
        plt.tight_layout()
        return fig


class AnomalyDetector(TimeSeriesAnalyzer):
    """시계열 이상 감지를 위한 클래스"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Parameters:
        -----------
        df : DataFrame
            분석할 데이터프레임
        """
        super().__init__(df)
        self.anomalies = None
        self.anomaly_methods = {}
    
    def detect_anomalies(self, method: str = 'iqr', window: int = None, 
                        threshold: float = 3.0, alpha: float = 0.05) -> pd.DataFrame:
        """시계열에서 이상치를 감지합니다.
        
        Parameters:
        -----------
        method : str, default='iqr'
            이상치 감지 방법 ('iqr', 'zscore', 'std', 'mad', 'prophet')
        window : int, default=None
            이동 창 크기 (None이면 전체 데이터 사용)
        threshold : float, default=3.0
            이상치 감지 임계값
        alpha : float, default=0.05
            통계적 검정을 위한 유의수준
        
        Returns:
        --------
        DataFrame
            이상치가 표시된 데이터프레임
        """
        if self.ts_df is None:
            raise ValueError("먼저 prepare_time_series 메서드를 호출하세요.")
        
        # 결측치 제거
        series = self.ts_df[self.value_column].dropna()
        
        # 이상치 감지
        if method == 'iqr':
            anomalies = self._detect_anomalies_iqr(series, threshold)
        elif method == 'zscore':
            anomalies = self._detect_anomalies_zscore(series, window, threshold)
        elif method == 'std':
            anomalies = self._detect_anomalies_std(series, window, threshold)
        elif method == 'mad':
            anomalies = self._detect_anomalies_mad(series, window, threshold)
        elif method == 'prophet':
            anomalies = self._detect_anomalies_prophet(series, threshold)
        else:
            raise ValueError(f"지원하지 않는 이상치 감지 방법: {method}")
        
        # 결과 저장
        self.anomalies = anomalies
        self.anomaly_methods[method] = anomalies
        
        return anomalies
    
    def _detect_anomalies_iqr(self, series: pd.Series, threshold: float = 1.5) -> pd.DataFrame:
        """IQR(Interquartile Range) 방법으로 이상치를 감지합니다."""
        # 사분위수 계산
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        
        # 이상치 경계 계산
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        
        # 이상치 감지
        is_anomaly = (series < lower_bound) | (series > upper_bound)
        
        # 결과 데이터프레임 생성
        anomalies = pd.DataFrame({
            self.value_column: series,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'is_anomaly': is_anomaly
        })
        
        return anomalies
    
    def _detect_anomalies_zscore(self, series: pd.Series, window: int = None, 
                              threshold: float = 3.0) -> pd.DataFrame:
        """Z-점수 방법으로 이상치를 감지합니다."""
        if window:
            # 이동 평균 및 표준편차 계산
            rolling_mean = series.rolling(window=window, center=True).mean()
            rolling_std = series.rolling(window=window, center=True).std()
            
            # 첫/마지막 부분의 누락된 값 처리
            rolling_mean = rolling_mean.fillna(series.mean())
            rolling_std = rolling_std.fillna(series.std())
            
            # Z-점수 계산
            z_scores = (series - rolling_mean) / rolling_std
        else:
            # 전체 데이터 기준 Z-점수
            z_scores = (series - series.mean()) / series.std()
        
        # 이상치 감지
        is_anomaly = abs(z_scores) > threshold
        
        # 경계 계산
        mean = series.mean() if not window else rolling_mean
        std = series.std() if not window else rolling_std
        lower_bound = mean - threshold * std
        upper_bound = mean + threshold * std
        
        # 결과 데이터프레임 생성
        anomalies = pd.DataFrame({
            self.value_column: series,
            'z_score': z_scores,
            'mean': mean,
            'std': std,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'is_anomaly': is_anomaly
        })
        
        return anomalies
    
    def _detect_anomalies_std(self, series: pd.Series, window: int = None, 
                          threshold: float = 3.0) -> pd.DataFrame:
        """표준편차 방법으로 이상치를 감지합니다."""
        if window:
            # 이동 평균 및 표준편차 계산
            rolling_mean = series.rolling(window=window, center=True).mean()
            rolling_std = series.rolling(window=window, center=True).std()
            
            # 첫/마지막 부분의 누락된 값 처리
            rolling_mean = rolling_mean.fillna(series.mean())
            rolling_std = rolling_std.fillna(series.std())
            
            # 경계 계산
            lower_bound = rolling_mean - threshold * rolling_std
            upper_bound = rolling_mean + threshold * rolling_std
        else:
            # 전체 데이터 기준 경계
            mean = series.mean()
            std = series.std()
            lower_bound = mean - threshold * std
            upper_bound = mean + threshold * std
            
            # 이동 평균/표준편차 설정
            rolling_mean = pd.Series([mean] * len(series), index=series.index)
            rolling_std = pd.Series([std] * len(series), index=series.index)
        
        # 이상치 감지
        is_anomaly = (series < lower_bound) | (series > upper_bound)
        
        # 결과 데이터프레임 생성
        anomalies = pd.DataFrame({
            self.value_column: series,
            'mean': rolling_mean,
            'std': rolling_std,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'is_anomaly': is_anomaly
        })
        
        return anomalies
    
    def _detect_anomalies_mad(self, series: pd.Series, window: int = None, 
                          threshold: float = 3.0) -> pd.DataFrame:
        """MAD(Median Absolute Deviation) 방법으로 이상치를 감지합니다."""
        if window:
            # 이동 중앙값 계산
            rolling_median = series.rolling(window=window, center=True).median()
            
            # 첫/마지막 부분의 누락된 값 처리
            rolling_median = rolling_median.fillna(series.median())
            
            # 이동 MAD 계산
            rolling_mad = np.abs(series - rolling_median).rolling(window=window, center=True).median()
            rolling_mad = rolling_mad.fillna(np.abs(series - series.median()).median())
        else:
            # 전체 데이터 기준 중앙값과 MAD
            median = series.median()
            mad = np.abs(series - median).median()
            
            # 이동 중앙값/MAD 설정
            rolling_median = pd.Series([median] * len(series), index=series.index)
            rolling_mad = pd.Series([mad] * len(series), index=series.index)
        
        # 일관된 표준편차 추정을 위한 보정 상수
        c = 1.4826  # 정규 분포 가정
        
        # 경계 계산
        lower_bound = rolling_median - threshold * c * rolling_mad
        upper_bound = rolling_median + threshold * c * rolling_mad
        
        # 이상치 감지
        is_anomaly = (series < lower_bound) | (series > upper_bound)
        
        # 결과 데이터프레임 생성
        anomalies = pd.DataFrame({
            self.value_column: series,
            'median': rolling_median,
            'mad': rolling_mad,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'is_anomaly': is_anomaly
        })
        
        return anomalies
    
    def _detect_anomalies_prophet(self, series: pd.Series, threshold: float = 0.99) -> pd.DataFrame:
        """Prophet 라이브러리를 사용하여 이상치를 감지합니다."""
        try:
            from prophet import Prophet
            from prophet.diagnostics import cross_validation
            from prophet.diagnostics import performance_metrics
            
            # Prophet 입력 형식 준비
            df_prophet = pd.DataFrame({
                'ds': series.index,
                'y': series.values
            })
            
            # Prophet 모델 초기화 및 학습
            model = Prophet(interval_width=threshold)
            model.fit(df_prophet)
            
            # 원본 기간에 대한 예측
            forecast = model.predict(df_prophet[['ds']])
            
            # 예측 구간 기반 이상치 감지
            is_anomaly = (
                (series.values < forecast['yhat_lower'].values) | 
                (series.values > forecast['yhat_upper'].values)
            )
            
            # 결과 데이터프레임 생성
            anomalies = pd.DataFrame({
                'ds': series.index,
                self.value_column: series.values,
                'yhat': forecast['yhat'].values,
                'lower_bound': forecast['yhat_lower'].values,
                'upper_bound': forecast['yhat_upper'].values,
                'is_anomaly': is_anomaly
            })
            
            anomalies.set_index('ds', inplace=True)
            
            return anomalies
            
        except ImportError:
            print("이상치 감지를 위해 Prophet 라이브러리가 필요합니다. pip install prophet")
            # 대체 방법 사용 (Z-점수)
            return self._detect_anomalies_zscore(series, None, threshold)
        except Exception as e:
            print(f"Prophet을 사용한 이상치 감지 중 오류 발생: {str(e)}")
            # 대체 방법 사용 (Z-점수)
            return self._detect_anomalies_zscore(series, None, threshold)
    
    def compare_anomaly_methods(self, methods: List[str] = None, 
                             window_sizes: List[int] = None) -> pd.DataFrame:
        """여러 이상치 감지 방법을 비교합니다.
        
        Parameters:
        -----------
        methods : list, default=None
            비교할 이상치 감지 방법 목록 (None이면 모든 방법)
        window_sizes : list, default=None
            비교할 이동 창 크기 목록
        
        Returns:
        --------
        DataFrame
            이상치 감지 방법 비교 결과
        """
        if self.ts_df is None:
            raise ValueError("먼저 prepare_time_series 메서드를 호출하세요.")
        
        if methods is None:
            methods = ['iqr', 'zscore', 'std', 'mad']  # Prophet은 계산 비용이 높아 제외
        
        if window_sizes is None:
            # 기본 창 크기 설정
            window_sizes = [None, 7, 30]
            
            # 시계열 길이에 따라 조정
            series_length = len(self.ts_df)
            if series_length < 10:
                window_sizes = [None]
            elif series_length < 30:
                window_sizes = [None, max(3, series_length // 3)]
            
        # 결과 저장
        results = []
        
        # 각 방법과 창 크기 조합에 대해 이상치 감지
        for method in methods:
            for window in window_sizes:
                # 'iqr' 방법은 창 크기를 사용하지 않음
                if method == 'iqr' and window is not None:
                    continue
                
                # Prophet은 창 크기를 사용하지 않음
                if method == 'prophet' and window is not None:
                    continue
                
                try:
                    # 이상치 감지
                    start_time = time.time()
                    anomalies = self.detect_anomalies(method=method, window=window)
                    end_time = time.time()
                    
                    # 결과 저장
                    result = {
                        'method': method,
                        'window': window if window is not None else 'None',
                        'anomaly_count': anomalies['is_anomaly'].sum(),
                        'anomaly_percentage': anomalies['is_anomaly'].mean() * 100,
                        'execution_time': end_time - start_time
                    }
                    
                    results.append(result)
                
                except Exception as e:
                    print(f"{method} 방법, 창 크기 {window}에서 오류 발생: {str(e)}")
        
        # 결과 데이터프레임 생성
        if results:
            results_df = pd.DataFrame(results)
            # 이상치 수 기준 정렬
            results_df = results_df.sort_values('anomaly_count', ascending=False)
            
            return results_df
        else:
            return pd.DataFrame(columns=['method', 'window', 'anomaly_count', 'anomaly_percentage', 'execution_time'])
    
    def get_anomaly_summary(self, method: str = None) -> Dict[str, Any]:
        """감지된 이상치의 요약 정보를 제공합니다.
        
        Parameters:
        -----------
        method : str, default=None
            요약할 이상치 감지 방법 (None이면 가장 최근 방법)
        
        Returns:
        --------
        dict
            이상치 요약 정보
        """
        if not self.anomaly_methods:
            raise ValueError("먼저 이상치를 감지하세요.")
        
        # 방법 선택
        if method is None:
            # 가장 최근에 사용한 방법
            method = list(self.anomaly_methods.keys())[-1]
        
        if method not in self.anomaly_methods:
            raise ValueError(f"'{method}' 방법으로 감지된 이상치가 없습니다.")
        
        anomalies = self.anomaly_methods[method]
        
        # 요약 정보 수집
        summary = {
            'method': method,
            'total_points': len(anomalies),
            'anomaly_count': anomalies['is_anomaly'].sum(),
            'anomaly_percentage': anomalies['is_anomaly'].mean() * 100
        }
        
        # 이상치 통계
        if summary['anomaly_count'] > 0:
            anomaly_points = anomalies[anomalies['is_anomaly']]
            normal_points = anomalies[~anomalies['is_anomaly']]
            
            summary['anomalies'] = {
                'values': {
                    'min': anomaly_points[self.value_column].min(),
                    'max': anomaly_points[self.value_column].max(),
                    'mean': anomaly_points[self.value_column].mean(),
                    'median': anomaly_points[self.value_column].median()
                },
                'dates': {
                    'first': anomaly_points.index.min(),
                    'last': anomaly_points.index.max()
                }
            }
            
            # 일별/주별/월별 이상치 분포
            if isinstance(anomalies.index, pd.DatetimeIndex):
                by_day = anomaly_points.groupby(anomaly_points.index.dayofweek).size()
                by_month = anomaly_points.groupby(anomaly_points.index.month).size()
                
                # 일/주/월/연도 별 이상치 비율
                day_of_week_ratio = anomalies.groupby(anomalies.index.dayofweek)['is_anomaly'].mean() * 100
                month_ratio = anomalies.groupby(anomalies.index.month)['is_anomaly'].mean() * 100
                
                summary['distribution'] = {
                    'by_day_of_week': by_day.to_dict(),
                    'by_month': by_month.to_dict(),
                    'day_of_week_ratio': day_of_week_ratio.to_dict(),
                    'month_ratio': month_ratio.to_dict()
                }
        
        return summary
    
    def plot_anomalies(self, method: str = None, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """감지된 이상치를 시각화합니다.
        
        Parameters:
        -----------
        method : str, default=None
            시각화할 이상치 감지 방법 (None이면 가장 최근 방법)
        figsize : tuple, default=(12, 6)
            그래프 크기
        
        Returns:
        --------
        matplotlib.figure.Figure
            이상치 시각화 그래프
        """
        if not self.anomaly_methods:
            raise ValueError("먼저 이상치를 감지하세요.")
        
        # 방법 선택
        if method is None:
            # 가장 최근에 사용한 방법
            method = list(self.anomaly_methods.keys())[-1]
        
        if method not in self.anomaly_methods:
            raise ValueError(f"'{method}' 방법으로 감지된 이상치가 없습니다.")
        
        anomalies = self.anomaly_methods[method]
        
        # 그래프 생성
        fig, ax = plt.subplots(figsize=figsize)
        
        # 원본 데이터 그리기
        ax.plot(anomalies.index, anomalies[self.value_column], 'b-', label='원본 데이터')
        
        # 이상치 그리기
        anomaly_points = anomalies[anomalies['is_anomaly']]
        ax.scatter(anomaly_points.index, anomaly_points[self.value_column], 
                  color='red', label='이상치', s=50, zorder=5)
        
        # 경계선 그리기 (있는 경우)
        if 'lower_bound' in anomalies.columns and 'upper_bound' in anomalies.columns:
            ax.plot(anomalies.index, anomalies['lower_bound'], 'g--', label='하한 경계')
            ax.plot(anomalies.index, anomalies['upper_bound'], 'g--', label='상한 경계')
            
            # 경계 영역 채우기
            ax.fill_between(
                anomalies.index,
                anomalies['lower_bound'],
                anomalies['upper_bound'],
                color='green', alpha=0.1
            )
        
        # 그래프 설정
        ax.set_title(f'{method} 방법으로 감지한 이상치')
        ax.set_xlabel('날짜')
        ax.set_ylabel(self.value_column)
        ax.legend()
        ax.grid(True)
        
        # 요약 정보 표시
        summary = self.get_anomaly_summary(method)
        summary_text = (
            f"총 데이터 수: {summary['total_points']}\n"
            f"이상치 수: {summary['anomaly_count']} ({summary['anomaly_percentage']:.2f}%)"
        )
        ax.text(0.05, 0.05, summary_text, transform=ax.transAxes, 
               bbox=dict(facecolor='white', alpha=0.7))
        
        plt.tight_layout()
        return fig