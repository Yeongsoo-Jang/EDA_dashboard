# utils/insights.py - 오늘의집 인사이트 생성 관련 함수
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from config import INSIGHT_THRESHOLDS, PRODUCT_CATEGORIES, BUSINESS_KPIS, USER_SEGMENTS

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

def generate_today_house_insights(df):
    """오늘의집 데이터에 특화된 인사이트를 생성합니다."""
    insights = generate_basic_insights(df)
    
    # 필수 열이 있는지 확인
    required_columns = ['order_date', 'total_price', 'category', 'user_id', 'product_id']
    missing_cols = [col for col in required_columns if col not in df.columns]
    
    if missing_cols:
        insights.append(f"⚠️ 일부 필수 열이 누락되어 인사이트 생성이 제한적입니다: {', '.join(missing_cols)}")
        return insights
    
    # 현재 시점 설정
    now = datetime.now()
    
    # 날짜 형식 확인 및 변환
    if not pd.api.types.is_datetime64_dtype(df['order_date']):
        try:
            df['order_date'] = pd.to_datetime(df['order_date'])
        except:
            insights.append("⚠️ 'order_date' 열을 날짜 형식으로 변환할 수 없어 시간 기반 인사이트를 생성할 수 없습니다.")
            return insights
    
    # 1. 매출 트렌드 분석
    try:
        # 월별 매출 집계
        monthly_sales = df.groupby(df['order_date'].dt.to_period('M'))['total_price'].sum()
        
        if len(monthly_sales) > 1:
            last_month = monthly_sales.index[-1]
            prev_month = monthly_sales.index[-2]
            
            change_pct = (monthly_sales[last_month] - monthly_sales[prev_month]) / monthly_sales[prev_month] * 100
            
            if change_pct > 10:
                insights.append(f"💰 최근 월 매출이 전월 대비 {change_pct:.1f}% 증가했습니다. 성장세가 강합니다.")
            elif change_pct > 5:
                insights.append(f"💰 최근 월 매출이 전월 대비 {change_pct:.1f}% 증가했습니다. 안정적인 성장을 보이고 있습니다.")
            elif change_pct < -10:
                insights.append(f"⚠️ 최근 월 매출이 전월 대비 {abs(change_pct):.1f}% 감소했습니다. 원인 분석이 필요합니다.")
            elif change_pct < -5:
                insights.append(f"⚠️ 최근 월 매출이 전월 대비 {abs(change_pct):.1f}% 감소했습니다. 주의 깊은 모니터링이 필요합니다.")
    except Exception as e:
        insights.append(f"⚠️ 매출 트렌드 분석 중 오류 발생: {str(e)}")
    
    # 2. 카테고리 인사이트
    try:
        category_sales = df.groupby('category')['total_price'].sum().sort_values(ascending=False)
        top_category = category_sales.index[0]
        top_category_pct = category_sales[top_category] / category_sales.sum() * 100
        
        # 최근 3개월 데이터
        three_months_ago = now - timedelta(days=90)
        recent_df = df[df['order_date'] >= three_months_ago]
        
        if not recent_df.empty:
            recent_category_sales = recent_df.groupby('category')['total_price'].sum().sort_values(ascending=False)
            recent_top_category = recent_category_sales.index[0]
            
            if recent_top_category != top_category:
                insights.append(f"📊 전체 기간에는 '{top_category}'이(가) 최고 매출 카테고리였으나, 최근 3개월간은 '{recent_top_category}'이(가) 가장 높은 매출을 기록하고 있습니다.")
            else:
                insights.append(f"📊 '{top_category}' 카테고리가 전체 매출의 {top_category_pct:.1f}%를 차지하며 지속적으로 인기를 유지하고 있습니다.")
        
        # 성장률이 가장 높은 카테고리 찾기
        if len(monthly_sales) > 3:  # 최소 3개월 데이터 필요
            # 최근 3개월과 이전 3개월 비교
            recent_3_months = df[df['order_date'] >= now - timedelta(days=90)]
            previous_3_months = df[(df['order_date'] < now - timedelta(days=90)) & 
                                    (df['order_date'] >= now - timedelta(days=180))]
            
            if not recent_3_months.empty and not previous_3_months.empty:
                recent_by_category = recent_3_months.groupby('category')['total_price'].sum()
                previous_by_category = previous_3_months.groupby('category')['total_price'].sum()
                
                # 두 기간 모두 데이터가 있는 카테고리만 비교
                common_categories = set(recent_by_category.index) & set(previous_by_category.index)
                
                growth_rates = {}
                for category in common_categories:
                    if previous_by_category[category] > 0:
                        growth_rate = (recent_by_category[category] - previous_by_category[category]) / previous_by_category[category] * 100
                        growth_rates[category] = growth_rate
                
                if growth_rates:
                    fastest_growing = max(growth_rates.items(), key=lambda x: x[1])
                    if fastest_growing[1] > 30:  # 30% 이상 성장한 경우만
                        insights.append(f"📈 '{fastest_growing[0]}' 카테고리가 전분기 대비 {fastest_growing[1]:.1f}% 성장하며 가장 빠른 성장세를 보이고 있습니다.")
                    
                    declining = [(cat, rate) for cat, rate in growth_rates.items() if rate < -20]
                    if declining:
                        declining.sort(key=lambda x: x[1])
                        insights.append(f"📉 '{declining[0][0]}' 카테고리가 전분기 대비 {abs(declining[0][1]):.1f}% 감소하며 가장 큰 하락세를 보이고 있습니다.")
    except Exception as e:
        insights.append(f"⚠️ 카테고리 인사이트 분석 중 오류 발생: {str(e)}")
    
    # 3. 고객 행동 인사이트
    try:
        # 재구매율 분석
        user_order_counts = df['user_id'].value_counts()
        repeat_buyers = user_order_counts[user_order_counts > 1].count()
        repeat_rate = repeat_buyers / user_order_counts.count() * 100
        
        insights.append(f"👥 전체 고객 중 {repeat_rate:.1f}%가 재구매를 한 충성 고객입니다.")
        
        # 최근 30일 신규 고객 비율
        recent_month = df[df['order_date'] >= now - timedelta(days=30)]
        if not recent_month.empty:
            recent_users = recent_month['user_id'].unique()
            existing_users = df[df['order_date'] < now - timedelta(days=30)]['user_id'].unique()
            new_users = set(recent_users) - set(existing_users)
            new_user_rate = len(new_users) / len(recent_users) * 100
            
            if new_user_rate > 30:
                insights.append(f"🆕 최근 30일 구매자 중 {new_user_rate:.1f}%가 신규 고객으로, 신규 유입이 활발합니다.")
            elif new_user_rate < 10:
                insights.append(f"⚠️ 최근 30일 구매자 중 신규 고객 비율이 {new_user_rate:.1f}%로 낮습니다. 신규 고객 유치 전략이 필요합니다.")
    except Exception as e:
        insights.append(f"⚠️ 고객 행동 인사이트 분석 중 오류 발생: {str(e)}")
    
    # 4. 가격대별 인사이트
    try:
        # 가격대 분류
        price_bins = [0, 10000, 30000, 50000, 100000, float('inf')]
        price_labels = ['1만원 미만', '1-3만원', '3-5만원', '5-10만원', '10만원 이상']
        
        df['price_range'] = pd.cut(df['total_price'], bins=price_bins, labels=price_labels)
        price_range_counts = df['price_range'].value_counts(normalize=True) * 100
        
        if '10만원 이상' in price_range_counts and price_range_counts['10만원 이상'] > 25:
            insights.append(f"💎 10만원 이상 고가 상품이 전체 주문의 {price_range_counts['10만원 이상']:.1f}%를 차지하여 고가 제품 비중이 높습니다.")
        elif '1만원 미만' in price_range_counts and price_range_counts['1만원 미만'] > 40:
            insights.append(f"🏷️ 1만원 미만 저가 상품이 전체 주문의 {price_range_counts['1만원 미만']:.1f}%를 차지합니다. 객단가 상승 전략을 고려해보세요.")
        
        # 객단가 트렌드
        if len(monthly_sales) > 1:
            monthly_orders = df.groupby(df['order_date'].dt.to_period('M'))['order_id'].count()
            monthly_aov = monthly_sales / monthly_orders
            
            last_month_aov = monthly_aov.iloc[-1]
            prev_month_aov = monthly_aov.iloc[-2]
            
            aov_change = (last_month_aov - prev_month_aov) / prev_month_aov * 100
            
            if aov_change > 10:
                insights.append(f"💰 최근 월 객단가가 {last_month_aov:,.0f}원으로 전월 대비 {aov_change:.1f}% 상승했습니다.")
            elif aov_change < -10:
                insights.append(f"⚠️ 최근 월 객단가가 {last_month_aov:,.0f}원으로 전월 대비 {abs(aov_change):.1f}% 하락했습니다.")
    except Exception as e:
        insights.append(f"⚠️ 가격대 인사이트 분석 중 오류 발생: {str(e)}")
    
    # 5. 계절성 및 시간 패턴 인사이트
    try:
        # 요일별 주문 패턴
        df['weekday'] = df['order_date'].dt.day_name()
        weekday_orders = df['weekday'].value_counts()
        top_weekday = weekday_orders.index[0]
        
        # 시간대별 주문 패턴
        df['hour'] = df['order_date'].dt.hour
        
        # 아침(6-10), 점심(11-14), 오후(15-18), 저녁(19-22), 밤(23-5)
        df['time_of_day'] = pd.cut(
            df['hour'], 
            bins=[-1, 5, 10, 14, 18, 22, 24], 
            labels=['밤', '아침', '점심', '오후', '저녁', '밤']
        )
        
        time_of_day_orders = df['time_of_day'].value_counts()
        peak_time = time_of_day_orders.index[0]
        
        insights.append(f"🕒 주문은 '{top_weekday}'요일과 '{peak_time}' 시간대에 가장 많이 발생합니다. 이 시간대를 타겟팅한 마케팅을 고려해보세요.")
        
        # 계절성 분석
        df['month'] = df['order_date'].dt.month
        month_orders = df.groupby('month')['order_id'].count()
        
        # 계절별 주문량
        seasons = {
            '봄': [3, 4, 5],
            '여름': [6, 7, 8],
            '가을': [9, 10, 11],
            '겨울': [12, 1, 2]
        }
        
        season_orders = {}
        for season, months in seasons.items():
            season_orders[season] = df[df['month'].isin(months)]['order_id'].count()
        
        top_season = max(season_orders.items(), key=lambda x: x[1])[0]
        bottom_season = min(season_orders.items(), key=lambda x: x[1])[0]
        
        season_ratio = season_orders[top_season] / season_orders[bottom_season]
        
        if season_ratio > 1.5:
            insights.append(f"🌈 '{top_season}'이 가장 주문이 많은 계절이며, '{bottom_season}'보다 {season_ratio:.1f}배 더 많은 주문이 발생합니다. 계절성이 뚜렷합니다.")
    except Exception as e:
        insights.append(f"⚠️ 시간 패턴 인사이트 분석 중 오류 발생: {str(e)}")
    
    # 6. 사용자 세그먼트 인사이트
    if 'user_segment' in df.columns:
        try:
            segment_counts = df['user_segment'].value_counts(normalize=True) * 100
            segment_sales = df.groupby('user_segment')['total_price'].sum()
            total_sales = segment_sales.sum()
            
            # 세그먼트별 매출 기여도
            segment_contribution = (segment_sales / total_sales * 100).sort_values(ascending=False)
            
            top_segment = segment_contribution.index[0]
            top_segment_contribution = segment_contribution[top_segment]
            
            segment_name = USER_SEGMENTS[top_segment]['name'] if top_segment in USER_SEGMENTS else top_segment
            
            insights.append(f"👑 '{segment_name}' 세그먼트가 전체 매출의 {top_segment_contribution:.1f}%를 차지하며 가장 가치 있는 고객층입니다.")
            
            # 비활성 고객 비율
            if 'at_risk_customers' in segment_counts:
                at_risk_pct = segment_counts['at_risk_customers']
                if at_risk_pct > 30:
                    insights.append(f"⚠️ 전체 고객 중 {at_risk_pct:.1f}%가 '이탈 위험' 상태로, 재활성화 캠페인이 필요합니다.")
        except Exception as e:
            insights.append(f"⚠️ 사용자 세그먼트 인사이트 분석 중 오류 발생: {str(e)}")
    
    # 7. 베스트 셀러 제품 인사이트
    try:
        product_sales = df.groupby('product_id')['total_price'].sum().sort_values(ascending=False)
        top_products = product_sales.head(5).index.tolist()
        
        if 'product_name' in df.columns:
            top_product_names = df[df['product_id'].isin(top_products)]['product_name'].unique()
            if len(top_product_names) > 0:
                insights.append(f"🏆 베스트셀러 상품은 '{top_product_names[0]}' 등으로, 이러한 인기 상품을 홍보 전략에 활용해보세요.")
    except Exception as e:
        insights.append(f"⚠️ 베스트셀러 인사이트 분석 중 오류 발생: {str(e)}")
    
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

def generate_kpi_insights(df):
    """비즈니스 KPI 관련 인사이트를 생성합니다."""
    insights = []
    
    # 필수 열이 있는지 확인
    required_columns = ['order_date', 'total_price', 'user_id', 'product_id']
    missing_cols = [col for col in required_columns if col not in df.columns]
    
    if missing_cols:
        insights.append(f"⚠️ 일부 필수 열이 누락되어 KPI 인사이트 생성이 제한적입니다: {', '.join(missing_cols)}")
        return insights
    
    try:
        # 현재 시점 설정 (데이터의 마지막 날짜를 현재로 간주)
        df['order_date'] = pd.to_datetime(df['order_date'])
        current_date = df['order_date'].max()
        
        # 최근 30일 데이터
        last_30_days = df[df['order_date'] >= current_date - timedelta(days=30)]
        
        # 이전 30일 데이터
        previous_30_days = df[(df['order_date'] < current_date - timedelta(days=30)) & 
                              (df['order_date'] >= current_date - timedelta(days=60))]
        
        # 1. 매출액 KPI
        current_revenue = last_30_days['total_price'].sum()
        previous_revenue = previous_30_days['total_price'].sum()
        
        if previous_revenue > 0:
            revenue_growth = (current_revenue - previous_revenue) / previous_revenue * 100
            target_growth = BUSINESS_KPIS['revenue']['target_increase'] * 100
            
            if revenue_growth >= target_growth:
                insights.append(f"🎯 최근 30일 매출액이 {current_revenue:,.0f}원으로 목표 성장률({target_growth:.1f}%)을 달성했습니다. (실제 성장률: {revenue_growth:.1f}%)")
            else:
                insights.append(f"📊 최근 30일 매출액은 {current_revenue:,.0f}원으로 목표 성장률({target_growth:.1f}%)에 미달합니다. (실제 성장률: {revenue_growth:.1f}%)")
        
        # 2. 전환율 KPI (가정: 방문 데이터가 없으므로 주문 건수 / 고유 사용자 수로 대체)
        current_orders = last_30_days['order_id'].nunique()
        current_users = last_30_days['user_id'].nunique()
        
        if current_users > 0:
            conversion_rate = current_orders / current_users * 100
            target_conversion = BUSINESS_KPIS['conversion_rate']['target_value']
            
            if conversion_rate >= target_conversion:
                insights.append(f"🎯 최근 30일 전환율이 {conversion_rate:.2f}%로 목표({target_conversion:.2f}%)를 달성했습니다.")
            else:
                gap = target_conversion - conversion_rate
                insights.append(f"📊 최근 30일 전환율은 {conversion_rate:.2f}%로 목표({target_conversion:.2f}%)까지 {gap:.2f}%p 남았습니다.")
        
        # 3. 객단가 KPI
        current_aov = current_revenue / current_orders if current_orders > 0 else 0
        target_aov = BUSINESS_KPIS['average_order_value']['target_value']
        
        if current_aov > 0:
            if current_aov >= target_aov:
                insights.append(f"🎯 최근 30일 객단가가 {current_aov:,.0f}원으로 목표({target_aov:,.0f}원)를 달성했습니다.")
            else:
                gap_percent = (target_aov - current_aov) / target_aov * 100
                insights.append(f"📊 최근 30일 객단가는 {current_aov:,.0f}원으로 목표({target_aov:,.0f}원)보다 {gap_percent:.1f}% 낮습니다.")
        
        # 4. 재구매율 KPI
        # 최근 30일 구매자 중 이전 30일에도 구매한 비율
        current_buyers = set(last_30_days['user_id'].unique())
        previous_buyers = set(previous_30_days['user_id'].unique())
        
        returning_buyers = current_buyers.intersection(previous_buyers)
        
        if current_buyers:
            retention_rate = len(returning_buyers) / len(current_buyers) * 100
            target_retention = BUSINESS_KPIS['retention_rate']['target_value']
            
            if retention_rate >= target_retention:
                insights.append(f"🎯 최근 30일 재구매율이 {retention_rate:.1f}%로 목표({target_retention:.1f}%)를 달성했습니다.")
            else:
                insights.append(f"📊 최근 30일 재구매율은 {retention_rate:.1f}%로 목표({target_retention:.1f}%)에 미달합니다.")
    
    except Exception as e:
        insights.append(f"⚠️ KPI 인사이트 생성 중 오류가 발생했습니다: {str(e)}")
    
    return insights