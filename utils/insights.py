# utils/insights.py - 인사이트 생성 관련 함수
import pandas as pd
import numpy as np
from config import INSIGHT_THRESHOLDS

def generate_basic_insights(df):
    """기본적인 데이터 인사이트를 생성합니다."""
    insights = []
    
    # 1. 결측치 분석
    missing_percentage = df.isnull().mean() * 100
    high_missing = missing_percentage[missing_percentage > INSIGHT_THRESHOLDS['high_missing']].index.tolist()
    
    if high_missing:
        insights.append(f"💡 다음 변수들은 결측치 비율이 {INSIGHT_THRESHOLDS['high_missing']}% 이상입니다: {', '.join(high_missing)}")
    
    # 2. 상관관계 분석
    numeric_df = df.select_dtypes(include=['number'])
    if not numeric_df.empty and len(numeric_df.columns) > 1:
        corr = numeric_df.corr().abs()
        # 중복 제거 및 자기 자신과의 상관관계 제거
        corr_pairs = corr.unstack().sort_values(ascending=False).drop_duplicates()
        high_corr_pairs = corr_pairs[(corr_pairs > INSIGHT_THRESHOLDS['high_correlation']) & (corr_pairs < 1.0)]
        
        if not high_corr_pairs.empty:
            top_5_pairs = high_corr_pairs.head(5)
            for idx, corr_value in top_5_pairs.items():
                var1, var2 = idx
                insights.append(f"💡 {var1}와 {var2} 간에 높은 상관관계가 있습니다 (상관계수: {corr_value:.2f})")
    
    # 3. 분포 분석 (왜도)
    for col in numeric_df.columns:
        if abs(numeric_df[col].skew()) > INSIGHT_THRESHOLDS['high_skew']:
            if numeric_df[col].skew() > 0:
                insights.append(f"💡 {col}은 오른쪽으로 치우친 분포를 보입니다 (왜도: {numeric_df[col].skew():.2f}). 로그 변환을 고려해 보세요.")
            else:
                insights.append(f"💡 {col}은 왼쪽으로 치우친 분포를 보입니다 (왜도: {numeric_df[col].skew():.2f}).")
    
    return insights

def generate_business_insights(df, data_type="generic"):
    """비즈니스 유형에 따른 맞춤형 인사이트를 생성합니다."""
    insights = generate_basic_insights(df)
    
    # 데이터 유형별 특화 인사이트
    if data_type == "sales":
        # 판매 데이터 관련 인사이트
        if '매출액' in df.columns and '날짜' in df.columns:
            # 시간별 매출 추세
            df['날짜'] = pd.to_datetime(df['날짜'])
            monthly_sales = df.groupby(df['날짜'].dt.to_period('M'))['매출액'].sum()
            
            if len(monthly_sales) > 1:
                last_month = monthly_sales.index[-1]
                prev_month = monthly_sales.index[-2]
                
                change_pct = (monthly_sales[last_month] - monthly_sales[prev_month]) / monthly_sales[prev_month] * 100
                
                if change_pct > 10:
                    insights.append(f"💡 최근 월 매출이 전월 대비 {change_pct:.1f}% 증가했습니다.")
                elif change_pct < -10:
                    insights.append(f"💡 최근 월 매출이 전월 대비 {abs(change_pct):.1f}% 감소했습니다. 원인 분석이 필요합니다.")
        
    elif data_type == "customer":
        # 고객 데이터 관련 인사이트
        if '연간지출액' in df.columns and '회원등급' in df.columns:
            # 회원등급별 지출 분석
            spending_by_tier = df.groupby('회원등급')['연간지출액'].mean().sort_values(ascending=False)
            
            top_tier = spending_by_tier.index[0]
            insights.append(f"💡 {top_tier} 회원의 평균 지출액이 {spending_by_tier[top_tier]:,.0f}원으로 가장 높습니다.")
    
    return insights

def generate_time_series_insights(df, date_column):
    """시계열 데이터에 대한 인사이트를 생성합니다."""
    insights = []
    
    if date_column in df.columns:
        # 날짜 형식으로 변환
        df['date'] = pd.to_datetime(df[date_column])
        
        # 수치형 변수에 대해 시계열 분석
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        for col in numeric_cols:
            # 시간에 따른 추세 (간단한 선형 회귀)
            if len(df) > 5:  # 최소 데이터 포인트 필요
                df['time_idx'] = range(len(df))
                corr = df[['time_idx', col]].corr().iloc[0, 1]
                
                if abs(corr) > 0.7:
                    trend_direction = "증가" if corr > 0 else "감소"
                    insights.append(f"💡 {col}은 시간에 따라 {trend_direction}하는 강한 추세를 보입니다 (상관계수: {corr:.2f}).")
    
    return insights