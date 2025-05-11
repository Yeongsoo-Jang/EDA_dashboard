# pages/home.py - 오늘의집 홈페이지 UI
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from utils.data_loader import generate_sample_data
from utils.insights import generate_today_house_insights, generate_kpi_insights
from config import BRAND_COLORS, BUSINESS_KPIS

def show_welcome():
    """시작 화면을 표시합니다."""
    st.title("📊 오늘의집 데이터 분석 대시보드")
    
    # 브랜드 색상 적용
    st.markdown(f"""
    <style>
    .main .block-container {{
        background-color: {BRAND_COLORS['background']};
    }}
    h1, h2, h3, h4, h5, h6 {{
        color: {BRAND_COLORS['text']};
    }}
    .stButton>button {{
        background-color: {BRAND_COLORS['primary']};
        color: white;
    }}
    .stButton>button:hover {{
        background-color: {BRAND_COLORS['tertiary']};
        color: white;
    }}
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### 환영합니다!
    
    이 대시보드는 오늘의집 비즈니스 데이터를 다양한 각도에서 분석하여 
    실용적인 인사이트를 제공합니다.
    
    **주요 기능:**
    - 📈 매출 및 주문 분석
    - 🛒 제품 카테고리별 성과 분석
    - 👥 고객 세그먼트 분석
    - 🔄 사용자 행동 패턴 분석
    - 🧠 고급 탐색적 데이터 분석 (EDA)
    - 🤖 머신러닝 기반 예측 모델링
    - 💡 자동화된 비즈니스 인사이트
    
    **시작하려면 왼쪽 사이드바에서 CSV 또는 JSON 파일을 업로드하거나
    아래에서 샘플 데이터를 선택하세요.**
    """)
    
    # 오늘의집 로고 (가상의 이미지 URL)
    logo_html = f"""
    <div style="display: flex; justify-content: center; margin: 2rem 0;">
        <div style="background-color: {BRAND_COLORS['primary']}; color: white; 
                   padding: 1rem 2rem; border-radius: 10px; font-size: 1.5rem; font-weight: bold;">
            오늘의집 데이터 분석
        </div>
    </div>
    """
    st.markdown(logo_html, unsafe_allow_html=True)
    
    # 샘플 데이터 옵션
    st.subheader("샘플 데이터로 시작하기")
    
    if st.button("📊 오늘의집 샘플 데이터 로드"):
        with st.spinner("샘플 데이터를 생성하는 중입니다..."):
            st.session_state.sample_file = generate_sample_data()
            st.success(f"{st.session_state.sample_file.name} 샘플 데이터가 로드되었습니다!")
            st.rerun()

def create_kpi_card(title, value, previous_value=None, format_str="{:,.0f}", unit="", target=None):
    """KPI 카드를 생성합니다."""
    formatted_value = format_str.format(value) + unit
    
    if previous_value is not None and previous_value != 0:
        change_pct = (value - previous_value) / previous_value * 100
        change_color = "green" if change_pct >= 0 else "red"
        change_icon = "↑" if change_pct >= 0 else "↓"
        change_text = f"{change_icon} {abs(change_pct):.1f}%"
    else:
        change_text = ""
        change_color = "gray"
    
    target_text = ""
    if target is not None:
        target_reached = value >= target
        target_color = "green" if target_reached else "orange"
        target_text = f"목표: {format_str.format(target)}{unit}"
    
    card_html = f"""
    <div style="background-color: white; border-radius: 10px; padding: 1rem; box-shadow: 0 2px 5px rgba(0,0,0,0.1); height: 100%;">
        <h4 style="color: {BRAND_COLORS['text']}; margin-top: 0; margin-bottom: 5px;">{title}</h4>
        <div style="font-size: 1.8rem; font-weight: bold; color: {BRAND_COLORS['text']};">{formatted_value}</div>
        <div style="display: flex; justify-content: space-between; margin-top: 10px;">
            <span style="color: {change_color};">{change_text}</span>
            <span style="color: {target_color if target_text else 'gray'}; font-size: 0.8rem;">{target_text}</span>
        </div>
    </div>
    """
    return card_html

def show(df, filename):
    """데이터 로드 후 홈페이지를 표시합니다."""
    # 브랜드 색상 적용
    st.markdown(f"""
    <style>
    .main .block-container {{
        background-color: {BRAND_COLORS['background']};
    }}
    h1, h2, h3, h4, h5, h6 {{
        color: {BRAND_COLORS['text']};
    }}
    .stButton>button {{
        background-color: {BRAND_COLORS['primary']};
        color: white;
    }}
    .stButton>button:hover {{
        background-color: {BRAND_COLORS['tertiary']};
        color: white;
    }}
    </style>
    """, unsafe_allow_html=True)
    
    st.title(f"📊 오늘의집 데이터 분석 - {filename}")
    
    # 이전 함수와 다음 컨텐츠 사이 간격 추가
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 날짜 열 변환 확인
    date_column = None
    for col in df.columns:
        if 'date' in col.lower() or '날짜' in col:
            try:
                df[col] = pd.to_datetime(df[col])
                date_column = col
                break
            except:
                pass
    
    # 대시보드 기간 필터 (날짜 열이 있는 경우)
    if date_column:
        min_date = df[date_column].min().date()
        max_date = df[date_column].max().date()
        
        with st.expander("📅 기간 필터", expanded=False):
            date_range = st.date_input(
                "분석 기간 선택",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
            
            if len(date_range) == 2:
                start_date, end_date = date_range
                filtered_df = df[(df[date_column].dt.date >= start_date) & (df[date_column].dt.date <= end_date)]
            else:
                filtered_df = df
        
        # 필터링된 데이터 정보
        period_text = f"기간: {date_range[0].strftime('%Y-%m-%d')} ~ {date_range[1].strftime('%Y-%m-%d')}" if len(date_range) == 2 else "전체 기간"
        st.markdown(f"**{period_text}** ({len(filtered_df)} 행)")
    else:
        filtered_df = df
    
    # KPI 섹션 - 주요 비즈니스 지표
    st.subheader("💼 주요 비즈니스 지표")
    
    # KPI 계산
    try:
        # 필수 열 확인
        required_columns = ['total_price', 'order_id', 'user_id', date_column]
        if all(col in filtered_df.columns for col in required_columns):
            # 기간 비교를 위한 데이터 분할
            if date_column:
                mid_date = filtered_df[date_column].max() - (filtered_df[date_column].max() - filtered_df[date_column].min()) / 2
                recent_data = filtered_df[filtered_df[date_column] >= mid_date]
                previous_data = filtered_df[filtered_df[date_column] < mid_date]
            else:
                # 날짜 열이 없는 경우 데이터를 절반으로 분할
                mid_point = len(filtered_df) // 2
                recent_data = filtered_df.iloc[mid_point:]
                previous_data = filtered_df.iloc[:mid_point]
            
            # KPI 계산
            # 1. 총 매출액
            current_revenue = recent_data['total_price'].sum()
            previous_revenue = previous_data['total_price'].sum()
            target_revenue = previous_revenue * (1 + BUSINESS_KPIS['revenue']['target_increase'])
            
            # 2. 주문 수
            current_orders = recent_data['order_id'].nunique()
            previous_orders = previous_data['order_id'].nunique()
            
            # 3. 객단가
            current_aov = current_revenue / current_orders if current_orders > 0 else 0
            previous_aov = previous_revenue / previous_orders if previous_orders > 0 else 0
            target_aov = BUSINESS_KPIS['average_order_value']['target_value']
            
            # 4. 고객 수
            current_customers = recent_data['user_id'].nunique()
            previous_customers = previous_data['user_id'].nunique()
            
            # KPI 카드 표시 (4개의 컬럼으로 구성)
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(create_kpi_card(
                    "총 매출액", 
                    current_revenue, 
                    previous_revenue, 
                    format_str="{:,.0f}", 
                    unit="원",
                    target=target_revenue
                ), unsafe_allow_html=True)
            
            with col2:
                st.markdown(create_kpi_card(
                    "주문 수", 
                    current_orders, 
                    previous_orders, 
                    format_str="{:,d}", 
                    unit="건"
                ), unsafe_allow_html=True)
            
            with col3:
                st.markdown(create_kpi_card(
                    "객단가", 
                    current_aov, 
                    previous_aov, 
                    format_str="{:,.0f}", 
                    unit="원",
                    target=target_aov
                ), unsafe_allow_html=True)
            
            with col4:
                st.markdown(create_kpi_card(
                    "고객 수", 
                    current_customers, 
                    previous_customers, 
                    format_str="{:,d}", 
                    unit="명"
                ), unsafe_allow_html=True)
        else:
            st.info("주요 비즈니스 지표를 계산하는 데 필요한 열이 없습니다.")
    except Exception as e:
        st.error(f"KPI 계산 중 오류가 발생했습니다: {str(e)}")
    
    # 데이터 미리보기 및 정보
    with st.expander("📋 데이터 미리보기 및 기본 정보", expanded=False):
        st.write(filtered_df.head())
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("행 수", filtered_df.shape[0])
            st.metric("결측치가 있는 열 수", filtered_df.isna().any().sum())
        with col2:
            st.metric("열 수", filtered_df.shape[1])
            st.metric("중복된 행 수", filtered_df.duplicated().sum())
        
        # 데이터 유형 표시
        st.subheader("데이터 유형")
        dtypes_df = pd.DataFrame(filtered_df.dtypes, columns=['데이터 유형'])
        dtypes_df['결측치 수'] = filtered_df.isnull().sum()
        dtypes_df['결측치 비율 (%)'] = (filtered_df.isnull().sum() / len(filtered_df) * 100).round(2)
        dtypes_df['고유값 수'] = filtered_df.nunique()
        st.dataframe(dtypes_df)
    
    # 주요 트렌드 차트 - 2개의 컬럼으로 구성
    st.subheader("📈 주요 트렌드")
    
    try:
        if date_column:
            col1, col2 = st.columns(2)
            
            with col1:
                # 일별/주별/월별 매출 트렌드
                if 'total_price' in filtered_df.columns:
                    trend_type = st.selectbox(
                        "시간 단위 선택", 
                        options=["일별", "주별", "월별"],
                        index=1,
                        key="trend_time_unit"
                    )
                    
                    if trend_type == "일별":
                        time_unit = 'D'
                        time_format = '%Y-%m-%d'
                    elif trend_type == "주별":
                        time_unit = 'W'
                        time_format = '%Y-%W'
                    else:
                        time_unit = 'M'
                        time_format = '%Y-%m'
                    
                    # 시간 단위별 데이터 집계
                    filtered_df['time_period'] = filtered_df[date_column].dt.to_period(time_unit)
                    time_series = filtered_df.groupby('time_period')['total_price'].sum().reset_index()
                    time_series['time_str'] = time_series['time_period'].astype(str)
                    
                    # 플롯리 차트
                    fig = px.line(
                        time_series, 
                        x='time_str', 
                        y='total_price',
                        labels={'time_str': '기간', 'total_price': '매출액'},
                        title=f"{trend_type} 매출 트렌드",
                        markers=True,
                        color_discrete_sequence=[BRAND_COLORS['primary']]
                    )
                    
                    fig.update_layout(
                        xaxis_title=f"{trend_type} 기간",
                        yaxis_title="매출액 (원)",
                        hovermode="x unified",
                        plot_bgcolor='white'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("매출액 데이터가 없습니다.")
            
            with col2:
                # 카테고리별 매출 분포
                if 'category' in filtered_df.columns and 'total_price' in filtered_df.columns:
                    category_sales = filtered_df.groupby('category')['total_price'].sum().sort_values(ascending=False)
                    
                    # 상위 8개 카테고리 + 기타
                    top_categories = category_sales.head(8)
                    if len(category_sales) > 8:
                        others_sum = category_sales[8:].sum()
                        top_categories = pd.concat([top_categories, pd.Series({'기타': others_sum})])
                    
                    fig = px.pie(
                        values=top_categories.values,
                        names=top_categories.index,
                        title="카테고리별 매출 비중",
                        color_discrete_sequence=BRAND_COLORS['categorical']
                    )
                    
                    fig.update_traces(
                        textposition='inside',
                        textinfo='percent+label',
                        hole=0.3,
                        marker=dict(line=dict(color='#FFFFFF', width=2))
                    )
                    
                    fig.update_layout(
                        legend_title="카테고리",
                        plot_bgcolor='white'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("카테고리 또는 매출 데이터가 없습니다.")
    except Exception as e:
        st.error(f"트렌드 차트 생성 중 오류가 발생했습니다: {str(e)}")
    
    # 고객 세그먼트 및 행동 분석
    st.subheader("👥 고객 세그먼트 및 행동 분석")
    
    try:
        col1, col2 = st.columns(2)
        
        with col1:
            # 사용자 세그먼트 분석
            if 'user_segment' in filtered_df.columns:
                segment_counts = filtered_df['user_segment'].value_counts()
                
                fig = px.bar(
                    x=segment_counts.index,
                    y=segment_counts.values,
                    title="고객 세그먼트 분포",
                    labels={'x': '세그먼트', 'y': '고객 수'},
                    color=segment_counts.index,
                    color_discrete_sequence=BRAND_COLORS['categorical']
                )
                
                fig.update_layout(
                    xaxis_title="세그먼트",
                    yaxis_title="고객 수",
                    plot_bgcolor='white',
                    bargap=0.3
                )
                
                st.plotly_chart(fig, use_container_width=True)
            elif 'user_id' in filtered_df.columns:
                # 사용자별 주문 빈도 분석
                order_frequency = filtered_df['user_id'].value_counts().value_counts().sort_index()
                
                fig = px.bar(
                    x=order_frequency.index,
                    y=order_frequency.values,
                    title="고객별 주문 빈도 분포",
                    labels={'x': '주문 횟수', 'y': '고객 수'},
                    color_discrete_sequence=[BRAND_COLORS['primary']]
                )
                
                fig.update_layout(
                    xaxis_title="주문 횟수",
                    yaxis_title="고객 수",
                    plot_bgcolor='white',
                    bargap=0.3
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("고객 세그먼트 또는 사용자 ID 데이터가 없습니다.")
        
        with col2:
            # 결제 방법 또는 배송 유형 분석
            if 'payment_method' in filtered_df.columns:
                payment_counts = filtered_df['payment_method'].value_counts()
                
                fig = px.pie(
                    values=payment_counts.values,
                    names=payment_counts.index,
                    title="결제 방법 분포",
                    color_discrete_sequence=BRAND_COLORS['categorical']
                )
                
                fig.update_traces(
                    textposition='inside',
                    textinfo='percent+label',
                    marker=dict(line=dict(color='#FFFFFF', width=1))
                )
                
                st.plotly_chart(fig, use_container_width=True)
            elif 'delivery_type' in filtered_df.columns:
                delivery_counts = filtered_df['delivery_type'].value_counts()
                
                fig = px.pie(
                    values=delivery_counts.values,
                    names=delivery_counts.index,
                    title="배송 유형 분포",
                    color_discrete_sequence=BRAND_COLORS['categorical']
                )
                
                fig.update_traces(
                    textposition='inside',
                    textinfo='percent+label',
                    marker=dict(line=dict(color='#FFFFFF', width=1))
                )
                
                st.plotly_chart(fig, use_container_width=True)
            elif 'region' in filtered_df.columns:
                # 지역별 주문 분포
                region_counts = filtered_df['region'].value_counts()
                
                # 상위 8개 지역 + 기타
                top_regions = region_counts.head(8)
                if len(region_counts) > 8:
                    others_sum = region_counts[8:].sum()
                    top_regions = pd.concat([top_regions, pd.Series({'기타': others_sum})])
                
                fig = px.pie(
                    values=top_regions.values,
                    names=top_regions.index,
                    title="지역별 주문 분포",
                    color_discrete_sequence=BRAND_COLORS['categorical']
                )
                
                fig.update_traces(
                    textposition='inside',
                    textinfo='percent+label',
                    marker=dict(line=dict(color='#FFFFFF', width=1))
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("결제 방법, 배송 유형 또는 지역 데이터가 없습니다.")
    except Exception as e:
        st.error(f"고객 분석 차트 생성 중 오류가 발생했습니다: {str(e)}")
    
    # 인사이트 섹션
    st.subheader("💡 주요 인사이트")
    
    try:
        # 오늘의집 데이터 인사이트 생성
        insights = generate_today_house_insights(filtered_df)
        kpi_insights = generate_kpi_insights(filtered_df)
        
        all_insights = insights + kpi_insights
        
        if all_insights:
            # 인사이트를 카테고리별로 분류
            sales_insights = [i for i in all_insights if any(keyword in i for keyword in ['매출', '객단가', '성장'])]
            customer_insights = [i for i in all_insights if any(keyword in i for keyword in ['고객', '세그먼트', '재구매'])]
            product_insights = [i for i in all_insights if any(keyword in i for keyword in ['카테고리', '제품', '상품'])]
            time_insights = [i for i in all_insights if any(keyword in i for keyword in ['시간', '요일', '계절'])]
            other_insights = [i for i in all_insights if i not in sales_insights + customer_insights + product_insights + time_insights]
            
            insight_tabs = st.tabs(["매출 인사이트", "고객 인사이트", "제품 인사이트", "시간 패턴", "기타 인사이트"])
            
            with insight_tabs[0]:
                if sales_insights:
                    for i in sales_insights:
                        st.markdown(i)
                else:
                    st.info("매출 관련 인사이트가 없습니다.")
            
            with insight_tabs[1]:
                if customer_insights:
                    for i in customer_insights:
                        st.markdown(i)
                else:
                    st.info("고객 관련 인사이트가 없습니다.")
            
            with insight_tabs[2]:
                if product_insights:
                    for i in product_insights:
                        st.markdown(i)
                else:
                    st.info("제품 관련 인사이트가 없습니다.")
            
            with insight_tabs[3]:
                if time_insights:
                    for i in time_insights:
                        st.markdown(i)
                else:
                    st.info("시간 패턴 관련 인사이트가 없습니다.")
            
            with insight_tabs[4]:
                if other_insights:
                    for i in other_insights:
                        st.markdown(i)
                else:
                    st.info("기타 인사이트가 없습니다.")
        else:
            st.info("데이터에서 특별한 인사이트를 발견하지 못했습니다.")
            
            
    except Exception as e:
        st.error(f"인사이트 생성 중 오류가 발생했습니다: {str(e)}")
    
    # 추천 분석 섹션
    st.subheader("🔍 권장 분석")
    
    recommended_analysis = [
        {
            "title": "카테고리별 성과 분석",
            "description": "각 제품 카테고리의 매출, 주문량, 평균 가격 등을 비교합니다.",
            "page": "변수 분석",
            "variables": "category"
        },
        {
            "title": "고객 세그먼트 분석",
            "description": "다양한 고객 세그먼트의 구매 행동과 선호도를 분석합니다.",
            "page": "고급 EDA",
            "variables": "user_segment, total_price"
        },
        {
            "title": "시간대별 주문 패턴",
            "description": "주문이 가장 많이 발생하는 시간대와 요일을 파악합니다.",
            "page": "변수 분석",
            "variables": f"{date_column}"
        },
        {
            "title": "제품 가격대별 분석",
            "description": "다양한 가격대의 제품 성과와 고객 선호도를 비교합니다.",
            "page": "변수 분석",
            "variables": "price, total_price"
        },
        {
            "title": "구매 예측 모델",
            "description": "고객의 다음 구매 가능성을 예측하는 머신러닝 모델을 구축합니다.",
            "page": "머신러닝 모델링",
            "variables": "user_id, order_count, days_since_last_order"
        }
    ]
    
    # 권장 분석 카드 표시
    cols = st.columns(3)
    for i, analysis in enumerate(recommended_analysis):
        with cols[i % 3]:
            st.markdown(f"""
            <div style="background-color: white; border-radius: 10px; padding: 1rem; 
                       box-shadow: 0 2px 5px rgba(0,0,0,0.1); height: 180px; margin-bottom: 1rem;">
                <h4 style="color: {BRAND_COLORS['text']}; margin-top: 0;">{analysis['title']}</h4>
                <p style="color: {BRAND_COLORS['text']}; font-size: 0.9rem; height: 60px;">{analysis['description']}</p>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="color: gray; font-size: 0.8rem;">페이지: {analysis['page']}</div>
                    <div style="color: gray; font-size: 0.8rem;">변수: {analysis['variables']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)