# pages/home.py - 오늘의집 홈페이지 UI
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from utils.data_loader import generate_sample_data # Assuming this is still needed for sample data
from utils.insights import *
# generate_today_house_insights, generate_kpi_insights, generate_data_quality_insights, generate_advanced_insights, generate_actionable_recommendations
from utils.data_processor import get_data_quality_report
from config import BRAND_COLORS, BUSINESS_KPIS, COLORSCALES

def show_welcome():    
    # 현재 테마에 맞는 색상 팔레트 가져오기
    current_theme_name = st.session_state.get("theme", "default")
    colors = BRAND_COLORS.get(current_theme_name, BRAND_COLORS["default"])

    # 브랜드 색상 적용
    st.markdown(f"""
    <style>
    .main .block-container {{
        background-color: {colors['background']};
    }}
    h1, h2, h3, h4, h5, h6 {{
        color: {colors['text']};
    }}
    .stButton>button {{
        background-color: {colors['primary']};
        color: white;
    }}
    .stButton>button:hover {{
        background-color: {colors['tertiary']};
        color: white;
    }}
    .stProgress > div > div {{
        background-color: {colors['primary']};
    }}
    </style>
    """, unsafe_allow_html=True)
    
    # 두 컬럼으로 분할
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        <h3 style="color: {colors['text']};">데이터 분석 대시보드에 오신 것을 환영합니다!</h3>
        
        이 대시보드는 비즈니스 데이터를 다양한 각도에서 분석하여
        실용적인 인사이트와 의사결정 지원을 제공합니다.
        
        **주요 기능:**
        - 📈 매출 및 주문 분석
        - 🛒 제품 카테고리별 성과 분석
        - 👥 고객 세그먼트 분석
        - 🔄 사용자 행동 패턴 분석
        - 🧠 고급 탐색적 데이터 분석 (EDA)
        - 🤖 머신러닝 기반 예측 모델링
        - 💡 실행 가능한 비즈니스 인사이트
        
        **시작하려면 왼쪽 사이드바에서 CSV, JSON 또는 엑셀 파일을 업로드하거나
        아래에서 샘플 데이터를 선택하세요.**
        """)

    with col2:
        # 오늘의집 로고/배너
        logo_html = f"""
        <div style="display: flex; justify-content: center; margin: 2rem 0;">
            <div style="background-color: {colors['primary']}; color: white; 
                    padding: 1.5rem; border-radius: 10px; text-align: center; width: 100%;">
                <div style="font-size: 2rem; font-weight: bold; margin-bottom: 0.5rem;">EDA</div>
                <div style="font-size: 1.2rem;">데이터 분석 대시보드</div>
            </div>
        </div>
        """
        st.markdown(logo_html, unsafe_allow_html=True)
        
        # 버전 정보 및 데이터 다운로드 링크
        st.markdown("""
        <div style="text-align: center; margin-top: 1rem; font-size: 0.8rem;">
            버전 1.1.0 | 2025년 5월 업데이트
        </div>
        """, unsafe_allow_html=True)
    
    # 샘플 데이터 옵션
    st.markdown(f"<h3 style='color: {colors['text']};'>샘플 데이터로 시작하기</h3>", unsafe_allow_html=True)
    
    # 샘플 데이터 카드 3개를 가로로 배치
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background-color: white; border-radius: 10px; padding: 1rem; height: 200px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
            <h4 style="color: {colors['text']};">📊 판매 데이터</h4>
            <p style="font-size: 0.9rem; color: #333333;">판매 및 주문 데이터 분석용 샘플</p>
            <p style="font-size: 0.8rem; color: #666666;">2,000+ 주문, 500+ 사용자</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("판매 데이터 로드", key="sales_data"):
            st.session_state.sample_file = generate_sample_data()
            if st.session_state.sample_file:
                st.session_state.current_df = st.session_state.sample_file[0]
                st.session_state.current_filename = st.session_state.sample_file[1]
                st.session_state.data_source_is_sample = True # Flag for app.py
                st.experimental_rerun() # Rerun app.py to use the sample data

            # st.success("오늘의집 샘플 데이터가 로드되었습니다! 사이드바에서 분석을 시작하세요.")
    
    with col2:
        st.markdown("""
        <div style="background-color: white; border-radius: 10px; padding: 1rem; height: 200px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
            <h4 style="color: {colors['text']};">👥 고객 데이터</h4>
            <p>오늘의집 고객 세그먼트 및 행동 데이터 샘플</p>
            <p style="font-size: 0.8rem; color: gray;">500+ 고객, 다양한 세그먼트</p>
        </div>
        """, unsafe_allow_html=True)
        # 실제 구현을 위해서는 고객 데이터 샘플 생성 함수 추가 필요
        if st.button("고객 데이터 로드", key="customer_data"):
            with st.spinner("고객 데이터 샘플을 생성하는 중입니다..."):
                st.warning("현재 고객 데이터 샘플은 준비 중입니다.")
    
    with col3:
        st.markdown("""
        <div style="background-color: white; border-radius: 10px; padding: 1rem; height: 200px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
            <h4 style="color: {colors['text']};">🏷️ 상품 데이터</h4>
            <p>오늘의집 상품 및 카테고리 데이터 샘플</p>
            <p style="font-size: 0.8rem; color: gray;">100+ 상품, 15+ 카테고리</p>
        </div>
        """, unsafe_allow_html=True)
        # 실제 구현을 위해서는 상품 데이터 샘플 생성 함수 추가 필요
        if st.button("상품 데이터 로드", key="product_data"):
            with st.spinner("상품 데이터 샘플을 생성하는 중입니다..."):
                st.warning("현재 상품 데이터 샘플은 준비 중입니다.")
    
    # 기능 소개 섹션
    st.markdown(f"<h3 style='color: {colors['text']};'>💫 주요 기능 소개</h3>", unsafe_allow_html=True)
    
    # 탭으로 기능 분류
    feature_tabs = st.tabs(["데이터 분석", "시각화", "머신러닝", "인사이트"])
    
    with feature_tabs[0]:
        st.markdown("""
        <h4 style="color: {colors['text']};">데이터 분석 기능</h4>
        
        - **기초 통계 분석**: 데이터의 기본 통계량 및 분포 확인
        - **변수별 상세 분석**: 각 변수의 특성과 영향력 분석
        - **상관관계 분석**: 변수 간 관계 및 패턴 탐색
        - **시계열 트렌드 분석**: 시간에 따른 데이터 변화 패턴 분석
        - **세그먼트 분석**: 고객 및 제품 세그먼트별 특성 비교
        """)

    with feature_tabs[1]:
        st.markdown("""
        <h4 style="color: {colors['text']};">시각화 기능</h4>
        
        - **인터랙티브 차트**: 마우스 오버로 세부 정보 확인
        - **다차원 시각화**: 3D 산점도, 레이더 차트로 복잡한 관계 표현
        - **지리적 분석**: 지역별 판매 및 고객 분포 시각화
        - **히트맵 & 상관관계 매트릭스**: 변수 간 관계 한눈에 파악
        - **시계열 차트**: 추세, 계절성, 이상치 시각화
        """)

    with feature_tabs[2]:
        st.markdown("""
        <h4 style="color: {colors['text']};">머신러닝 기능</h4>
        
        - **예측 모델링**: 회귀/분류 모델로 미래 예측
        - **고객 세분화**: 자동 군집화로 고객 그룹 발견
        - **구매 확률 예측**: 고객별 다음 구매 확률 계산
        - **상품 추천**: 사용자 행동 기반 개인화 추천
        - **이탈 예측**: 고객 이탈 가능성 분석 및 예방
        """)

    with feature_tabs[3]:
        st.markdown("""
        <h4 style="color: {colors['text']};">인사이트 기능</h4>
        
        - **자동 KPI 추적**: 주요 비즈니스 지표 모니터링
        - **이상 감지**: 데이터 이상치 및 특이 패턴 발견
        - **트렌드 알림**: 주요 변화 및 추세 자동 감지
        - **실행 가능한 제안**: 데이터 기반 비즈니스 의사결정 지원
        - **보고서 생성**: 분석 결과를 PDF로 내보내기
        """)

    # 푸터
    st.markdown("""
    <div style="margin-top: 3rem; text-align: center; color: gray; font-size: 0.8rem;">
        © 2025 오늘의집 데이터 분석팀 | 문의: data-team@ohouse.com
    </div>
    """, unsafe_allow_html=True)
    
def create_kpi_card(title, value, colors, previous_value=None, format_str="{:,.0f}", unit="", target=None, icon=None):
    """향상된 KPI 카드를 생성합니다."""
    formatted_value = format_str.format(value) + unit
    
    # 변화율 계산 및 스타일 지정
    if previous_value is not None and previous_value != 0:
        change_pct = (value - previous_value) / previous_value * 100
        change_color = "#2C8D80" if change_pct >= 0 else "#FF6B6B"
        change_icon = "↑" if change_pct >= 0 else "↓"
        change_text = f"{change_icon} {abs(change_pct):.1f}%"
    else:
        change_text = ""
        change_color = "#8e8e8e"
    
    # 목표 텍스트 및 스타일
    target_text = ""
    target_color = "#8e8e8e"
    if target is not None:
        target_reached = value >= target
        target_color = "#2C8D80" if target_reached else "#FF9F1C"
        target_text = f"목표: {format_str.format(target)}{unit}"
    
    # 아이콘 결정
    if icon is None:
        icon_map = {
            "매출": "💰", "총 매출액": "💰", "revenue": "💰",
            "주문": "📦", "주문 수": "📦", "orders": "📦",
            "객단가": "💎", "average": "💎", "aov": "💎",
            "고객": "👥", "고객 수": "👥", "customers": "👥",
            "제품": "🏷️", "제품 수": "🏷️", "products": "🏷️",
            "방문": "👀", "방문자 수": "👀", "visitors": "👀",
        }
        
        for key, symbol in icon_map.items():
            if key in title.lower():
                icon = symbol
                break
        
        # 기본 아이콘
        if icon is None:
            icon = "📊"
    
    # 향상된 카드 HTML - 모든 텍스트에 명시적 색상 지정으로 가독성 보장
    card_html = f"""
    <div style="background-color: white; border-radius: 10px; padding: 1rem; box-shadow: 0 2px 5px rgba(0,0,0,0.1); height: 100%;">
        <div style="display: flex; align-items: center; margin-bottom: 8px;">
            <div style="font-size: 1.8rem; margin-right: 8px;">{icon}</div>
            <h4 style="color: {colors['text']}; margin: 0;">{title}</h4>
        </div>
        <div style="font-size: 1.8rem; font-weight: bold; color: {colors['text']}; margin: 10px 0;">{formatted_value}</div>
        <div style="display: flex; justify-content: space-between; margin-top: 10px;">
            <span style="color: {change_color}; font-weight: bold;">{change_text}</span>
            <span style="color: {target_color}; font-size: 0.8rem;">{target_text}</span>
        </div>
    </div>
    """
    return card_html

def show(df, filename):
    """데이터 로드 후 홈페이지를 표시합니다."""
    # 현재 테마에 맞는 색상 팔레트 가져오기
    current_theme_name = st.session_state.get("theme", "default")
    colors = BRAND_COLORS.get(current_theme_name, BRAND_COLORS["default"])

    # 브랜드 색상 적용
    st.markdown(f"""
    <style>
    .main .block-container {{
        background-color: {colors['background']};
    }}
    h1, h2, h3, h4, h5, h6 {{
        color: {colors['text']};
    }}
    .stButton>button {{
        background-color: {colors['primary']};
        color: white;
    }}
    .stButton>button:hover {{
        background-color: {colors['tertiary']};
        color: white;
    }}
    .stProgress > div > div {{
        background-color: {colors['primary']};
    }}
    /* 카드 호버 효과 */
    div[data-testid="stHorizontalBlock"] > div:hover {{
        transform: translateY(-5px);
        transition: transform 0.3s ease;
    }}
    /* 탭 스타일링 */
    button[data-baseweb="tab"] {{
        font-size: 1rem !important; /* Streamlit 기본 스타일 오버라이드를 위해 !important 추가 */
        font-weight: 600;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        color: {BRAND_COLORS['primary']} !important;
        border-bottom-color: {BRAND_COLORS['primary']} !important;
    }}
    </style>
    """, unsafe_allow_html=True)
    
    # 상단 헤더
    st.title(f"📊 오늘의집 데이터 분석")
    st.markdown(f"<h4 style='margin-top: -10px; color: {colors['text']}; opacity: 0.8;'>{filename} 분석 결과</h4>", 
               unsafe_allow_html=True)
    
    # 이전 함수와 다음 컨텐츠 사이 간격 추가
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 데이터 품질 점수 계산
    quality_report = get_data_quality_report(df)
    missing_percentage = quality_report["missing_percentage"]
    duplicate_percentage = quality_report["duplicate_rows"] / quality_report["row_count"] * 100 if quality_report["row_count"] > 0 else 0
    
    # 날짜 열 변환 확인
    date_column = None
    for col in df.columns:
        if pd.api.types.is_datetime64_dtype(df[col]):
            date_column = col
            break
    
    # 대시보드 기간 필터 (날짜 열이 있는 경우)
    if date_column:
        min_date = df[date_column].min().date()
        max_date = df[date_column].max().date()
        
        with st.expander("📅 기간 필터", expanded=False):
            date_filter_type = st.radio(
                "필터 유형 선택:",
                ["전체 기간", "특정 기간", "최근 기간"]
            )
            
            if date_filter_type == "특정 기간":
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
                    
                # 필터링된 기간 정보
                period_text = f"분석 기간: {date_range[0].strftime('%Y-%m-%d')} ~ {date_range[1].strftime('%Y-%m-%d')}" if len(date_range) == 2 else "전체 기간"
            
            elif date_filter_type == "최근 기간":
                period_options = {
                    "최근 7일": 7,
                    "최근 30일": 30,
                    "최근 90일": 90,
                    "최근 180일": 180,
                    "최근 1년": 365
                }
                selected_period = st.selectbox("기간 선택:", list(period_options.keys()))
                days = period_options[selected_period]
                
                cutoff_date = max_date - timedelta(days=days)
                filtered_df = df[df[date_column].dt.date > cutoff_date]
                period_text = f"분석 기간: {selected_period} ({cutoff_date.strftime('%Y-%m-%d')} ~ {max_date.strftime('%Y-%m-%d')})"
            
            else:
                filtered_df = df
                period_text = f"전체 기간: {min_date.strftime('%Y-%m-%d')} ~ {max_date.strftime('%Y-%m-%d')}"
    else:
        filtered_df = df
        period_text = "전체 데이터"
    
    # 품질 점수 (100점 만점)
    quality_score = 100
    
    # 결측치에 따른 감점 (최대 -30점)
    if missing_percentage > 0:
        quality_score -= min(30, missing_percentage * 3)
    
    # 중복 행에 따른 감점 (최대 -20점)
    if duplicate_percentage > 0:
        quality_score -= min(20, duplicate_percentage * 4)
    
    # 부적절한 데이터 유형에 따른 감점 (최대 -20점)
    inappropriate_types = 0
    for col_info in quality_report["columns"]:
        # 날짜 관련 열이 문자열인 경우
        if any(date_keyword in col_info["column_name"].lower() for date_keyword in ['date', 'time', '날짜', '일자']) and col_info["data_type"] == 'object':
            inappropriate_types += 1
        # 금액 관련 열이 문자열인 경우
        elif any(price_keyword in col_info["column_name"].lower() for price_keyword in ['price', 'amount', 'cost', 'revenue', '금액', '가격']) and col_info["data_type"] == 'object':
            inappropriate_types += 1
    
    if inappropriate_types > 0:
        quality_score -= min(20, inappropriate_types * 5)
    
    # 이상치에 따른 감점 (최대 -30점)
    outlier_penalty = 0
    for col_info in quality_report["columns"]:
        if "outliers_percentage" in col_info and col_info["outliers_percentage"] > 10:
            outlier_penalty += min(10, col_info["outliers_percentage"] * 0.5)
    
    quality_score -= min(30, outlier_penalty)
    
    # 최종 점수 (0-100 사이로 조정)
    quality_score = max(0, min(100, quality_score))
    
    # 품질 등급 결정
    if quality_score >= 90:
        quality_grade = "A+"
        quality_color = "#2C8D80"
    elif quality_score >= 80:
        quality_grade = "A"
        quality_color = "#3DBFAD"
    elif quality_score >= 70:
        quality_grade = "B+"
        quality_color = "#50E3C2"
    elif quality_score >= 60:
        quality_grade = "B"
        quality_color = "#66D9E8"
    elif quality_score >= 50:
        quality_grade = "C+"
        quality_color = "#FFD43B"
    elif quality_score >= 40:
        quality_grade = "C"
        quality_color = "#FF9F1C"
    else:
        quality_grade = "D"
        quality_color = "#FF6B6B"
    
    # 상단 요약 정보
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # 데이터 요약
        st.markdown(f"""
        <div style="background-color: white; border-radius: 10px; padding: 1rem; box-shadow: 0 2px 5px rgba(0,0,0,0.1); height: 150px;">
            <h4 style="margin-top: 0; margin-bottom: 10px; color: {colors['text']};">📋 데이터 개요</h4>
            <p style="margin: 5px 0;"><strong>행:</strong> {len(filtered_df):,}</p>
            <p style="margin: 5px 0;"><strong>열:</strong> {len(filtered_df.columns):,}</p>
            <p style="margin: 5px 0;"><strong>{period_text}</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # 데이터 품질 점수
        st.markdown(f"""
        <div style="background-color: white; border-radius: 10px; padding: 1rem; box-shadow: 0 2px 5px rgba(0,0,0,0.1); height: 150px;">
            <h4 style="margin-top: 0; margin-bottom: 10px; color: {colors['text']};">✅ 데이터 품질</h4>
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <span style="font-size: 2rem; font-weight: bold; color: {quality_color};">{quality_grade}</span>
                <span style="font-size: 1.5rem; font-weight: bold; color: {quality_color};">{quality_score:.1f}/100</span>
            </div>
            <div style="margin-top: 10px; font-size: 0.9rem;">
                <span>결측치: {missing_percentage:.1f}%</span>
                <span style="float: right;">중복: {duplicate_percentage:.1f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # 분석 준비도
        # 분석에 필요한 핵심 열이 있는지 확인
        required_columns_check = ['order_id', 'user_id', 'product_id', 'price', 'total_price', 'category']
        available_columns = [col for col in required_columns_check if col in filtered_df.columns]
        readiness_score = len(available_columns) / len(required_columns_check) * 100
        
        readiness_text = "높음 ✅" if readiness_score >= 80 else "중간 ⚠️" if readiness_score >= 50 else "낮음 ❌"
        readiness_color = "#2C8D80" if readiness_score >= 80 else "#FF9F1C" if readiness_score >= 50 else "#FF6B6B"
        
        st.markdown(f"""
        <div style="background-color: white; border-radius: 10px; padding: 1rem; box-shadow: 0 2px 5px rgba(0,0,0,0.1); height: 150px;">
            <h4 style="margin-top: 0; margin-bottom: 10px; color: {colors['text']};">🔍 분석 준비도</h4>
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <span style="font-size: 1.2rem; font-weight: bold; color: {readiness_color};">{readiness_text}</span>
                <span style="margin-left: auto; font-weight: bold; color: {readiness_color};">{readiness_score:.0f}%</span>
            </div>
            <div style="height: 10px; background-color: #f0f0f0; border-radius: 5px; overflow: hidden;">
                <div style="height: 100%; width: {readiness_score}%; background-color: {readiness_color};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        # 데이터 시간 범위 (날짜 열이 있는 경우)
        if date_column:
            date_range_days = (filtered_df[date_column].max() - filtered_df[date_column].min()).days
            
            st.markdown(f"""
            <div style="background-color: white; border-radius: 10px; padding: 1rem; box-shadow: 0 2px 5px rgba(0,0,0,0.1); height: 150px;">
                <h4 style="margin-top: 0; margin-bottom: 10px; color: {colors['text']};">📅 시간 범위</h4>
                <div style="font-size: 1.5rem; font-weight: bold; color: {colors['text']}; margin-bottom: 10px;">{date_range_days}일</div>
                <p style="margin: 5px 0; font-size: 0.9rem;">시작: {filtered_df[date_column].min().strftime('%Y-%m-%d')}</p>
                <p style="margin: 5px 0; font-size: 0.9rem;">종료: {filtered_df[date_column].max().strftime('%Y-%m-%d')}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background-color: white; border-radius: 10px; padding: 1rem; box-shadow: 0 2px 5px rgba(0,0,0,0.1); height: 150px;">
                <h4 style="margin-top: 0; margin-bottom: 10px; color: {colors['text']};">⚠️ 날짜 정보 없음</h4>
                <p>시간 기반 분석을 위해 날짜 열이 필요합니다.</p>
                <p style="font-size: 0.9rem;">시간 분석 기능이 제한됩니다.</p>
            </div>
            """, unsafe_allow_html=True)
    
    # KPI 섹션 - 주요 비즈니스 지표
    st.subheader(f"💼 주요 비즈니스 지표")
    
    # KPI 계산
    try:
        # 필수 열 확인
        required_kpi_columns = ['total_price', 'order_id', 'user_id', date_column]
        if all(col in filtered_df.columns for col in required_kpi_columns if col is not None):
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
            
            # 5. 구매 전환율 (가정)
            if 'category' in filtered_df.columns:
                product_count = filtered_df['category'].nunique() # A proxy for browsed products
                conversion_rate = current_orders / (current_customers * product_count) * 100 if (current_customers * product_count) > 0 else 0
                previous_conversion = previous_orders / (previous_customers * product_count) * 100 if (previous_customers * product_count) > 0 else 0
                target_conversion = BUSINESS_KPIS['conversion_rate']['target_value']
            else:
                conversion_rate = 0
                previous_conversion = 0
                target_conversion = 0
            
            # KPI 카드 표시 (5개의 컬럼으로 구성)
            kpi_cols = st.columns(5)
            col1, col2, col3, col4, col5 = kpi_cols
            
            with col1:
                st.markdown(create_kpi_card(
                    "총 매출액", 
                    current_revenue, 
                    previous_revenue, 
                    format_str="{:,.0f}", 
                    colors=colors,
                    unit="원",
                    target=target_revenue,
                    icon="💰"
                ), unsafe_allow_html=True)
            
            with col2:
                st.markdown(create_kpi_card(
                    "주문 수", 
                    current_orders, 
                    previous_orders, 
                    format_str="{:,d}", 
                    colors=colors,
                    unit="건",
                    icon="📦"
                ), unsafe_allow_html=True)
            
            with col3:
                st.markdown(create_kpi_card(
                    "객단가", 
                    current_aov, 
                    previous_aov, 
                    format_str="{:,.0f}", 
                    colors=colors,
                    unit="원",
                    target=target_aov,
                    icon="💎"
                ), unsafe_allow_html=True)
            
            with col4:
                st.markdown(create_kpi_card(
                    "고객 수", 
                    current_customers, 
                    previous_customers, 
                    format_str="{:,d}", 
                    unit="명",
                    colors=colors
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
    st.subheader(f"📈 주요 트렌드")
    
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
                        markers=True
                    )
                    
                    fig.update_layout(
                        xaxis_title=f"{trend_type} 기간",
                        yaxis_title="매출액 (원)",
                        hovermode="x unified",
                        plot_bgcolor='white'
                    )                    
                    fig.update_traces(line=dict(color=colors['primary']))
                    
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
                        color_discrete_sequence=COLORSCALES['categorical']
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
    st.subheader(f"👥 고객 세그먼트 및 행동 분석")
    
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
                    color_discrete_sequence=COLORSCALES['categorical']
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
                    color_discrete_sequence=[colors['primary']]
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
                    color_discrete_sequence=COLORSCALES['categorical']
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
                    color_discrete_sequence=COLORSCALES['categorical']
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
                    color_discrete_sequence=COLORSCALES['categorical']
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
    st.subheader(f"💡 주요 인사이트")
    
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
    st.subheader(f"🔍 권장 분석")
    
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
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1); height: 200px; margin-bottom: 1rem; display: flex; flex-direction: column; justify-content: space-between;">
                <div><h4 style="color: {colors['text']}; margin-top: 0;">{analysis['title']}</h4>
                <p style="color: {colors['text']}; font-size: 0.9rem; height: 60px;">{analysis['description']}</p></div>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="color: {colors['text']}; opacity: 0.7; font-size: 0.8rem;">페이지: {analysis['page']}</div>
                    <div style="color: {colors['text']}; opacity: 0.7; font-size: 0.8rem;">변수: {analysis['variables']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)