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
    
    # 파일 업로드 섹션
    st.subheader("📁 데이터 업로드")
    uploaded_file = st.file_uploader("CSV, JSON 또는 엑셀 파일을 업로드하세요", 
                                     type=['csv', 'json', 'xlsx', 'xls'],
                                     help="데이터 분석을 위한 파일을 업로드하세요.")
    
    # 로딩 상태 표시 개선
    if uploaded_file is not None:
        with st.status("데이터 처리 중...", expanded=True) as status:
            st.write("파일 로드 중...")
            # load_data now returns a tuple: (DataFrame, redundant_cols_info)
            actual_df, redundant_info = load_data(uploaded_file)
            
            st.session_state['data'] = actual_df # Store the DataFrame part
            st.session_state['filename'] = uploaded_file.name
            # Store redundant_info for later use (e.g., UI for removing columns)
            st.session_state['redundant_cols_info_for_ui'] = redundant_info
            
            # 데이터 정보 요약 표시
            if actual_df is not None:
                st.write(f"✅ 데이터 로드 완료: {actual_df.shape[0]:,}행 × {actual_df.shape[1]:,}열")
                memory_usage = actual_df.memory_usage(deep=True).sum() / (1024 * 1024)
                st.write(f"📊 메모리 사용량: {memory_usage:.2f} MB")
                
                # 대용량 데이터 감지 및 경고
                if actual_df.shape[0] * actual_df.shape[1] > APP_CONFIG["large_data_threshold"]:
                    st.warning("대용량 데이터가 감지되었습니다. 일부 분석은 시간이 오래 걸릴 수 있습니다.")
                status.update(label="데이터 로드 완료!", state="complete")
            else:
                status.update(label="데이터 로드 실패.", state="error")
                st.session_state['data'] = None # Ensure data is None if actual_df is None
                st.session_state['redundant_cols_info_for_ui'] = []
    else:
        # 샘플 데이터 옵션
        # 이 블록은 st.file_uploader를 통해 파일이 업로드되지 않았을 때 실행됩니다.
        if 'sample_file' in st.session_state and st.session_state.sample_file is not None:
            # st.session_state.sample_file이 (DataFrame, 파일명) 튜플이라고 가정합니다.
            # 이 값은 home.py 등에서 generate_sample_data() 호출 후 설정됩니다.
            with st.status("샘플 데이터 적용 중...") as status:
                sample_data_tuple = st.session_state.sample_file

                # st.session_state.sample_file이 튜플(DataFrame, 파일명)인지 확인
                if isinstance(sample_data_tuple, tuple) and \
                   len(sample_data_tuple) == 2 and \
                   isinstance(sample_data_tuple[0], pd.DataFrame) and \
                   isinstance(sample_data_tuple[1], str):
                    
                    df_sample, filename_sample = sample_data_tuple
                    
                    if df_sample is not None:
                        st.session_state['data'] = df_sample
                        st.session_state['filename'] = filename_sample
                        st.session_state["preprocessing_applied"] = False # 새 데이터 로드시 전처리 플래그 초기화
                        status.update(label="샘플 데이터 적용 완료!", state="complete")
                    else:
                        st.error("샘플 데이터의 DataFrame이 비어있습니다.")
                        status.update(label="샘플 데이터 적용 실패!", state="error")
                        st.session_state['data'] = None
                        st.session_state['filename'] = None
                else:
                    st.error("세션의 샘플 데이터 형식이 올바르지 않습니다. (DataFrame, 파일명) 튜플이어야 합니다.")
                    status.update(label="샘플 데이터 형식 오류!", state="error")
                    st.session_state['data'] = None
                    st.session_state['filename'] = None
        else:
            st.session_state['data'] = None
            st.session_state['filename'] = None # filename도 None으로 설정


    
    # 사이드바 - 여러 파일 업로드
    st.sidebar.subheader("📁 여러 파일 업로드")
    uploaded_files = st.sidebar.file_uploader(
        "CSV 파일을 여러 개 업로드하세요", type=['csv'], accept_multiple_files=True
    )

    # 업로드된 파일을 데이터프레임으로 읽기
    if uploaded_files:
        st.sidebar.write(f"업로드된 파일 수: {len(uploaded_files)}")
        dataframes = {file.name: pd.read_csv(file) for file in uploaded_files}
        
        # 파일 미리보기
        st.sidebar.subheader("파일 미리보기")
        selected_file = st.sidebar.selectbox("파일 선택", list(dataframes.keys()))
        if selected_file:
            st.sidebar.write(dataframes[selected_file].head())

        # 병합 옵션
        st.subheader("🛠️ 데이터 병합")
        merge_method = st.radio("병합 방법 선택", ["SQL 쿼리", "Python 코드"])
        
        if merge_method == "SQL 쿼리":
            st.text_area("SQL 쿼리를 입력하세요 (예: SELECT * FROM df1 JOIN df2 ON df1.id = df2.id)", key="sql_query")
            if st.button("SQL 실행"):
                try:
                    merged_data = ps.sqldf(st.session_state.sql_query, locals())
                    st.session_state["merged_data"] = merged_data
                    st.success("SQL 쿼리 실행 완료!")
                    st.write(merged_data.head())
                except Exception as e:
                    st.error(f"SQL 실행 중 오류 발생: {e}")
        
        elif merge_method == "Python 코드":
            st.text_area("Python 코드를 입력하세요 (예: df1.merge(df2, on='id'))", key="python_code")
            if st.button("Python 코드 실행"):
                try:
                    exec(st.session_state.python_code, globals(), locals())
                    merged_data = locals().get("merged_data")
                    if merged_data is not None:
                        st.session_state["merged_data"] = merged_data
                        st.success("Python 코드 실행 완료!")
                        st.write(merged_data.head())
                    else:
                        st.error("`merged_data` 변수를 생성해야 합니다.")
                except Exception as e:
                    st.error(f"Python 코드 실행 중 오류 발생: {e}")

    # 병합된 데이터가 있으면 대시보드에서 분석 가능
    if "merged_data" in st.session_state:
        st.subheader("📊 병합된 데이터 분석")
        st.write(st.session_state["merged_data"].head())
        # 기존 대시보드 페이지로 연결
        page = st.radio(
            "분석 페이지 선택",
            ["홈", "기초 통계", "변수 분석", "고급 EDA", "머신러닝 모델링"],
            index=0
        )
        if page == "홈":
            home.show(st.session_state["merged_data"], "병합된 데이터")
        elif page == "기초 통계":
            basic_stats_page.show(st.session_state["merged_data"])
        elif page == "변수 분석":
            variable_page.show(st.session_state["merged_data"])
        elif page == "고급 EDA":
            advanced_page.show(st.session_state["merged_data"])
        elif page == "머신러닝 모델링":
            ml_page.show(st.session_state["merged_data"])

    # 데이터 전처리 옵션 (새로 추가)
    if st.session_state['data'] is not None and not st.session_state["preprocessing_applied"]:
        st.subheader("⚙️ 데이터 전처리")
        
        # Define checkboxes and button inside the expander
        # Their state will be captured when the script runs
        with st.expander("전처리 옵션", expanded=False):
            # Use unique keys for widgets to ensure their state is managed correctly
            handle_missing_option = st.checkbox("결측치 처리", value=True, key="cb_handle_missing")
            remove_duplicates_option = st.checkbox("중복 행 제거", value=True, key="cb_remove_duplicates")
            normalize_columns_option = st.checkbox("열 이름 정규화", value=True, key="cb_normalize_columns")
            apply_button_clicked = st.button("전처리 적용", key="btn_apply_preprocessing")

        # If the button was clicked, execute the preprocessing and show status.
        # This 'if' block is now a sibling to the 'with st.expander(...)' block,
        # so st.status is no longer nested within st.expander.
        if apply_button_clicked:
            with st.status("전처리 중...") as status:
                df = st.session_state['data']
                # Use the captured checkbox values for preprocessing
                df = preprocess_data(df, {
                    "handle_missing": handle_missing_option,
                    "remove_duplicates": remove_duplicates_option,
                    "normalize_columns": normalize_columns_option
                })
                st.session_state['data'] = df
                st.session_state["preprocessing_applied"] = True
                status.update(label="전처리 완료!", state="complete")
                st.rerun()

    # 페이지 선택(데이터 병합 모듈)
    st.subheader("📑 페이지 선택")
    page = st.radio(
        "",
        ["홈", "기초 통계", "변수 분석", "고급 EDA", "머신러닝 모델링", "데이터 병합"],
        index=0,
        format_func=lambda x: {
            "홈": "🏠 홈",
            "기초 통계": "📊 기초 통계",
            "변수 분석": "📈 변수 분석",
            "고급 EDA": "🧠 고급 EDA",
            "머신러닝 모델링": "🤖 머신러닝 모델링",
            "데이터 병합": "🔄 데이터 병합"  # 새로 추가된 페이지
        }[x]
    )

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

# 메인 콘텐츠 - 페이지에 따른 내용 표시
if 'data' in st.session_state and st.session_state['data'] is not None:
    # 인메모리 캐싱을 위한 세션 상태 활용
    if page == "홈":
        home.show(st.session_state['data'], st.session_state['filename'])
    elif page == "기초 통계":
        basic_stats_page.show(st.session_state['data'])
    elif page == "변수 분석":
        variable_page.show(st.session_state['data'])
    elif page == "고급 EDA":
        advanced_page.show(st.session_state['data'])
    elif page == "머신러닝 모델링":
        ml_page.show(st.session_state['data'])
    elif page == "데이터 병합":
        # 새로 추가된 데이터 병합 모듈 호출
        from modules.data_merger import show as show_data_merger
        show_data_merger()
else:
    # 데이터가 없을 때 샘플 데이터 옵션 제공
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