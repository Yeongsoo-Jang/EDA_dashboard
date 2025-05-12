# app.py - 메인 애플리케이션 진입점
import streamlit as st
import pandas as pd
from modules import home, basic_stats_page, variable_page, advanced_page, ml_page
from utils.data_loader import load_data, generate_sample_data
from utils.data_processor import preprocess_data
from config import BRAND_COLORS, APP_CONFIG
import pandasql as ps  # SQL 쿼리 실행을 위한 라이브러리


# 페이지 설정 - 반응형 디자인 지원
st.set_page_config(
    page_title="EDA 데이터 분석 대시보드",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 테마 설정 및 반응형 스타일 적용
def apply_theme(theme=None):
    """테마를 적용합니다. 기본은 'default'입니다."""
    theme = theme or "default"
    colors = BRAND_COLORS.get(theme, BRAND_COLORS["default"])
    
    st.markdown(f"""
    <style>
        /* 기본 스타일 */
        .main .block-container {{
            background-color: {colors['background']};
            padding: clamp(1rem, 2vw, 2rem);
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: {colors['text']};
        }}
        
        /* 버튼 스타일 */
        .stButton>button {{
            background-color: {colors['primary']};
            color: white;
            transition: all 0.3s ease;
        }}
        .stButton>button:hover {{
            background-color: {colors['tertiary']};
            color: white;
            transform: translateY(-2px);
        }}
        
        /* 프로그레스 바 스타일 */
        .stProgress > div > div {{
            background-color: {colors['primary']};
        }}
        
        /* 사이드바 스타일 */
        [data-testid="stSidebar"] {{
            background-color: {colors['tertiary']};
        }}
        [data-testid="stSidebar"] .sidebar-content {{
            color: white;
        }}
        [data-testid="stSidebarUserContent"] {{
            padding-top: 1rem;
        }}
        
        /* 카드 스타일 */
        div[data-testid="stHorizontalBlock"] > div {{
            transition: transform 0.3s ease;
        }}
        div[data-testid="stHorizontalBlock"] > div:hover {{
            transform: translateY(-5px);
        }}
        
        /* 반응형 디자인 */
        @media (max-width: 768px) {{
            .responsive-flex {{
                flex-direction: column !important;
            }}
            .card {{
                margin-bottom: 1rem !important;
            }}
        }}
        
        /* 탭 스타일 */
        button[data-baseweb="tab"] {{
            font-size: 1rem;
            font-weight: 600;
        }}
        button[data-baseweb="tab"][aria-selected="true"] {{
            color: {colors['primary']} !important;
            border-bottom-color: {colors['primary']} !important;
        }}
        
        /* 푸터 숨기기 */
        footer {{
            display: none !important;
        }}
    </style>
    """, unsafe_allow_html=True)

# 테마 적용
apply_theme(st.session_state.get("theme", "default"))

# 세션 상태 초기화
if "data" not in st.session_state:
    st.session_state["data"] = None
if "filename" not in st.session_state:
    st.session_state["filename"] = None
if "preprocessing_applied" not in st.session_state:
    st.session_state["preprocessing_applied"] = False
if "active_dataframes" not in st.session_state:
    st.session_state.active_dataframes = {}
if "merged_results" not in st.session_state:
    st.session_state.merged_results = {}
if "show_merger" not in st.session_state:
    st.session_state.show_merger = False
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "홈"

# 사이드바 - 페이지 네비게이션
with st.sidebar:
    st.title("📊 데이터 분석 대시보드")
    
    # 로고 - 사이드바 상단
    colors = BRAND_COLORS.get(st.session_state.get("theme", "default"), BRAND_COLORS["default"])
    logo_html = f"""
    <div style="text-align: center; margin-bottom: 30px; margin-top: 10px;">
        <div style="background-color: white; color: {colors['primary']}; 
                   padding: 1rem; border-radius: 10px; font-size: 1.2rem; font-weight: bold;">
            데이터 분석 대시보드
        </div>
    </div>
    """
    st.markdown(logo_html, unsafe_allow_html=True)
    
    # 통합 파일 업로드 섹션
    st.subheader("📁 데이터 업로드")
    
    upload_type = st.radio(
        "업로드 유형:",
        ["단일 파일", "다중 파일"],
        horizontal=True,
        help="분석용 단일 파일 또는 병합용 다중 파일을 선택하세요."
    )
    
    if upload_type == "단일 파일":
        uploaded_file = st.file_uploader(
            "CSV, JSON 또는 엑셀 파일을 업로드하세요",
            type=['csv', 'json', 'xlsx', 'xls'],
            help="단일 파일 분석을 위해 업로드하세요."
        )
        
        # 단일 파일 처리 로직
        if uploaded_file is not None:
            with st.status("데이터 처리 중...", expanded=True) as status:
                st.write("파일 로드 중...")
                data_df, redundant_info = load_data(uploaded_file)
                
                if data_df is not None:
                    st.session_state['data'] = data_df
                    st.session_state['filename'] = uploaded_file.name
                    st.session_state['redundant_cols_info_for_ui'] = redundant_info
                    st.session_state.active_dataframes[uploaded_file.name] = data_df
                    
                    st.write(f"✅ 데이터 로드 완료: {data_df.shape[0]:,}행 × {data_df.shape[1]:,}열")
                    memory_usage = data_df.memory_usage(deep=True).sum() / (1024 * 1024)
                    st.write(f"📊 메모리 사용량: {memory_usage:.2f} MB")
                    
                    # 대용량 데이터 감지 및 경고
                    if data_df.shape[0] * data_df.shape[1] > APP_CONFIG["large_data_threshold"]:
                        st.warning("대용량 데이터가 감지되었습니다. 일부 분석은 시간이 오래 걸릴 수 있습니다.")
                    status.update(label="데이터 로드 완료!", state="complete")
                else:
                    status.update(label="데이터 로드 실패.", state="error")
                    st.session_state['data'] = None # Ensure data is None if actual_df is None
                    st.session_state['redundant_cols_info_for_ui'] = []
    
    else:  # 다중 파일
        uploaded_files = st.file_uploader(
            "여러 파일을 업로드하세요",
            type=['csv', 'json', 'xlsx', 'xls'],
            accept_multiple_files=True,
            help="데이터 병합을 위해 여러 파일을 업로드하세요."
        )
        
        # 다중 파일 처리 로직
        if uploaded_files:
            with st.status("다중 파일 처리 중...") as status:
                progress_bar = st.progress(0)
                
                for i, file in enumerate(uploaded_files):
                    progress_bar.progress((i + 0.5) / len(uploaded_files))
                    st.write(f"파일 로드 중: {file.name}")
                    
                    try:
                        # 파일 로드
                        data_df, _ = load_data(file)
                        if data_df is not None:
                            # 파일명에서 확장자 제거
                            file_name = '.'.join(file.name.split('.')[:-1])
                            # 이미 같은 이름의 파일이 있는 경우 처리
                            if file_name in st.session_state.active_dataframes:
                                file_name = f"{file_name}_{len(st.session_state.active_dataframes)}"
                                
                            st.session_state.active_dataframes[file_name] = data_df
                    except Exception as e:
                        st.error(f"파일 '{file.name}' 로드 중 오류 발생: {str(e)}")
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                st.success(f"{len(uploaded_files)}개 파일 로드 완료!")
                status.update(label="모든 파일 로드 완료!", state="complete")
    
    # 샘플 데이터 로직
    if st.session_state['data'] is None and 'sample_file' in st.session_state and st.session_state.sample_file is not None:
        with st.status("샘플 데이터 적용 중...") as status:
            sample_data_tuple = st.session_state.sample_file

            if isinstance(sample_data_tuple, tuple) and \
               len(sample_data_tuple) == 2 and \
               isinstance(sample_data_tuple[0], pd.DataFrame) and \
               isinstance(sample_data_tuple[1], str):
                
                df_sample, filename_sample = sample_data_tuple
                
                if df_sample is not None:
                    st.session_state['data'] = df_sample
                    st.session_state['filename'] = filename_sample
                    st.session_state["preprocessing_applied"] = False
                    # 활성 데이터프레임에도 추가
                    st.session_state.active_dataframes[filename_sample] = df_sample
                    status.update(label="샘플 데이터 적용 완료!", state="complete")
                else:
                    st.error("샘플 데이터의 DataFrame이 비어있습니다.")
                    status.update(label="샘플 데이터 적용 실패!", state="error")
            else:
                st.error("세션의 샘플 데이터 형식이 올바르지 않습니다.")
                status.update(label="샘플 데이터 형식 오류!", state="error")
    
    # 데이터 병합 버튼 - 다중 파일이 업로드된 경우에만 표시
    if len(st.session_state.active_dataframes) >= 2:
        st.subheader("🔄 데이터 병합")
        if st.button("데이터 병합 시작", use_container_width=True):
            st.session_state["show_merger"] = True
            st.session_state["merge_step"] = 1  # 병합 단계 초기화
            st.rerun()
    
    # 데이터 전처리 옵션
    if st.session_state['data'] is not None and not st.session_state["preprocessing_applied"]:
        st.subheader("⚙️ 데이터 전처리")
        
        with st.expander("전처리 옵션", expanded=False):
            handle_missing_option = st.checkbox("결측치 처리", value=True, key="cb_handle_missing")
            remove_duplicates_option = st.checkbox("중복 행 제거", value=True, key="cb_remove_duplicates")
            normalize_columns_option = st.checkbox("열 이름 정규화", value=True, key="cb_normalize_columns")
            apply_button_clicked = st.button("전처리 적용", key="btn_apply_preprocessing")

        if apply_button_clicked:
            with st.status("전처리 중...") as status:
                df = st.session_state['data']
                df = preprocess_data(df, {
                    "handle_missing": handle_missing_option,
                    "remove_duplicates": remove_duplicates_option,
                    "normalize_columns": normalize_columns_option
                })
                st.session_state['data'] = df
                st.session_state["preprocessing_applied"] = True
                status.update(label="전처리 완료!", state="complete")
                st.rerun()

    # 페이지 선택 네비게이션
    st.subheader("📑 페이지 선택")
    if "data" in st.session_state and st.session_state["data"] is not None:
        page = st.radio(
            "",
            ["홈", "기초 통계", "변수 분석", "고급 EDA", "머신러닝 모델링"],
            format_func=lambda x: {
                "홈": "🏠 홈",
                "기초 통계": "📊 기초 통계",
                "변수 분석": "📈 변수 분석",
                "고급 EDA": "🧠 고급 EDA",
                "머신러닝 모델링": "🤖 머신러닝 모델링"
            }[x]
        )
        st.session_state["current_page"] = page
    else:
        st.info("데이터를 업로드하여 분석을 시작하세요.")

    # 푸터
    st.markdown("---")
    st.markdown("""
    <div style="color: white; opacity: 0.7; font-size: 0.8rem; text-align: center; margin-top: 50px;">
        © 2025 데이터 분석 대시보드<br>
        버전 2.0.0
    </div>
    """, unsafe_allow_html=True)

    # 가이드 정보
    with st.expander("ℹ️ 사용 가이드"):
        st.markdown("""
        - **홈**: 데이터 개요 및 주요 KPI 확인
        - **기초 통계**: 기본 통계량 및 변수별 분포
        - **변수 분석**: 개별 변수 심층 분석 및 관계 탐색
        - **고급 EDA**: PCA, 군집화, 3D 시각화 등
        - **머신러닝 모델링**: 예측 모델 구축 및 평가
        """)

# 메인 콘텐츠 영역
if "show_merger" in st.session_state and st.session_state["show_merger"]:
    # 병합 UI 표시
    from modules.data_merger import show_simplified_merger
    show_simplified_merger()
    
    # 메인 페이지로 돌아가기 버튼
    if st.button("← 메인 페이지로 돌아가기", key="back_to_main"):
        st.session_state["show_merger"] = False
        st.rerun()
        
elif "data" in st.session_state and st.session_state["data"] is not None:
    # 현재 선택된 페이지 표시
    current_page = st.session_state.get("current_page", "홈")
    
    if current_page == "홈":
        home.show(st.session_state['data'], st.session_state['filename'])
    elif current_page == "기초 통계":
        basic_stats_page.show(st.session_state['data'])
    elif current_page == "변수 분석":
        variable_page.show(st.session_state['data'])
    elif current_page == "고급 EDA":
        advanced_page.show(st.session_state['data'])
    elif current_page == "머신러닝 모델링":
        ml_page.show(st.session_state['data'])
else:
    # 데이터가 없을 때 웰컴 페이지
    home.show_welcome()

# 성능 모니터링 (개발자 모드)
if st.session_state.get('dev_mode', False):
    st.sidebar.subheader("🔧 개발자 옵션")
    if st.sidebar.checkbox("성능 모니터링 표시"):
        st.sidebar.code("""
        # 메모리 사용량
        Current Memory Usage: {} MB
        """.format(
            round(pd.DataFrame().memory_usage(deep=True).sum() / (1024 * 1024), 2)
            if 'data' not in st.session_state or st.session_state['data'] is None
            else round(st.session_state['data'].memory_usage(deep=True).sum() / (1024 * 1024), 2)
        ))