# modules/data_merger.py - 고급 데이터 병합 모듈
import streamlit as st
import pandas as pd
import numpy as np
import sqlparse
import networkx as nx
import matplotlib.pyplot as plt
from io import StringIO
import json
import time
import re
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go

class DataMerger:
    """고급 데이터 병합 기능을 제공하는 클래스"""

    def __init__(self):
        """병합 모듈 초기화"""
        self.dataframes = {}
        self.merge_history = []
        self.sql_templates = {
            "기본 조인": "SELECT a.*, b.* FROM df1 a JOIN df2 b ON a.id = b.id",
            "왼쪽 조인": "SELECT a.*, b.* FROM df1 a LEFT JOIN df2 b ON a.id = b.id",
            "집계 쿼리": "SELECT category, SUM(value) as total FROM df1 GROUP BY category",
            "다중 테이블 조인": "SELECT a.*, b.*, c.* FROM df1 a JOIN df2 b ON a.id = b.id JOIN df3 c ON b.id = c.id"
        }
        self.merge_presets = {}
        
        # 세션 상태 초기화
        if 'merged_results' not in st.session_state:
            st.session_state.merged_results = {}
        if 'active_dataframes' not in st.session_state:
            st.session_state.active_dataframes = {}
        if 'merge_history' not in st.session_state:
            st.session_state.merge_history = []
        if 'last_error' not in st.session_state:
            st.session_state.last_error = None

    def show(self):
        """병합 모듈 UI 표시"""
        st.title("🔄 고급 데이터 병합")
        
        # 상단 설명
        st.markdown("""
        이 모듈을 사용하여 여러 CSV, Excel 또는 JSON 파일을 병합하세요. 
        직관적인 인터페이스 또는 SQL 쿼리를 통해 복잡한 데이터 병합을 수행할 수 있습니다.
        """)
        
        # 메인 탭 구성
        main_tabs = st.tabs(["파일 업로드", "스키마 탐색", "병합 도구", "결과 관리"])
        
        with main_tabs[0]:
            self._show_file_upload()
            
        with main_tabs[1]:
            self._show_schema_explorer()
            
        with main_tabs[2]:
            self._show_merge_tools()
            
        with main_tabs[3]:
            self._show_results_manager()

    def _show_file_upload(self):
        """파일 업로드 UI"""
        st.header("파일 업로드")
        
        upload_cols = st.columns([2, 1])
        
        with upload_cols[0]:
            uploaded_files = st.file_uploader(
                "CSV, Excel 또는 JSON 파일을 업로드하세요",
                type=['csv', 'xlsx', 'xls', 'json'],
                accept_multiple_files=True,
                help="여러 파일을 한 번에 업로드하거나 개별적으로 추가할 수 있습니다."
            )
            
            if uploaded_files:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, file in enumerate(uploaded_files):
                    try:
                        status_text.text(f"파일 로드 중... ({i+1}/{len(uploaded_files)}): {file.name}")
                        progress_bar.progress((i + 0.5) / len(uploaded_files))
                        
                        # 파일 확장자 확인
                        file_extension = file.name.split('.')[-1].lower()
                        
                        # 파일 타입에 따른 로드
                        if file_extension == 'csv':
                            try:
                                # UTF-8 먼저 시도
                                df = pd.read_csv(file)
                            except UnicodeDecodeError:
                                # UTF-8 실패시 cp949 시도
                                file.seek(0)
                                df = pd.read_csv(file, encoding='cp949')
                        elif file_extension in ['xlsx', 'xls']:
                            df = pd.read_excel(file)
                        elif file_extension == 'json':
                            df = pd.read_json(file)
                        else:
                            st.warning(f"지원하지 않는 파일 형식: {file_extension}")
                            continue
                        
                        # 데이터프레임 저장 (파일명에서 확장자 제거)
                        file_name = '.'.join(file.name.split('.')[:-1])
                        # 이미 같은 이름의 파일이 있는 경우 처리
                        if file_name in st.session_state.active_dataframes:
                            file_name = f"{file_name}_{len(st.session_state.active_dataframes)}"
                            
                        st.session_state.active_dataframes[file_name] = df
                        
                        progress_bar.progress((i + 1) / len(uploaded_files))
                    
                    except Exception as e:
                        st.error(f"파일 '{file.name}' 로드 중 오류 발생: {str(e)}")
                
                progress_bar.empty()
                status_text.success(f"{len(uploaded_files)}개 파일 로드 완료!")
                
        with upload_cols[1]:
            st.subheader("샘플 데이터")
            if st.button("샘플 데이터 로드", help="테스트용 샘플 데이터셋을 로드합니다."):
                with st.spinner("샘플 데이터 생성 중..."):
                    # 판매 데이터 샘플
                    sales_df = pd.DataFrame({
                        'order_id': [f"ORD-{i}" for i in range(1001, 1011)],
                        'customer_id': [f"CUST-{i}" for i in range(101, 111)],
                        'product_id': [f"PROD-{i}" for i in range(1, 11)],
                        'order_date': pd.date_range(start='2023-01-01', periods=10),
                        'quantity': np.random.randint(1, 10, size=10),
                        'total_price': np.random.uniform(100, 1000, size=10).round(2)
                    })
                    
                    # 고객 데이터 샘플
                    customer_df = pd.DataFrame({
                        'customer_id': [f"CUST-{i}" for i in range(101, 111)],
                        'name': [f"고객 {i}" for i in range(1, 11)],
                        'city': np.random.choice(['서울', '부산', '인천', '대구', '광주'], size=10),
                        'age': np.random.randint(20, 60, size=10),
                        'join_date': pd.date_range(start='2022-01-01', periods=10)
                    })
                    
                    # 제품 데이터 샘플
                    product_df = pd.DataFrame({
                        'product_id': [f"PROD-{i}" for i in range(1, 11)],
                        'product_name': [f"제품 {i}" for i in range(1, 11)],
                        'category': np.random.choice(['전자제품', '의류', '식품', '가구', '도서'], size=10),
                        'price': np.random.uniform(50, 500, size=10).round(2),
                        'stock': np.random.randint(0, 100, size=10)
                    })
                    
                    # 데이터프레임 저장
                    st.session_state.active_dataframes['sales'] = sales_df
                    st.session_state.active_dataframes['customers'] = customer_df
                    st.session_state.active_dataframes['products'] = product_df
                    
                    st.success("샘플 데이터가 로드되었습니다: 'sales', 'customers', 'products'")
        
        # 현재 로드된 데이터프레임 목록
        st.subheader("로드된 데이터셋")
        if st.session_state.active_dataframes:
            data_tabs = st.tabs(list(st.session_state.active_dataframes.keys()))
            
            for i, (name, df) in enumerate(st.session_state.active_dataframes.items()):
                with data_tabs[i]:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.dataframe(df.head(5), use_container_width=True)
                    with col2:
                        st.write(f"**행:** {df.shape[0]}")
                        st.write(f"**열:** {df.shape[1]}")
                        
                        # 데이터셋 삭제 버튼
                        if st.button("데이터셋 삭제", key=f"delete_{name}"):
                            del st.session_state.active_dataframes[name]
                            st.rerun()
        else:
            st.info("파일을 업로드하거나 샘플 데이터를 로드하세요.")

    def _show_schema_explorer(self):
        """스키마 탐색 UI"""
        st.header("스키마 탐색")
        
        if not st.session_state.active_dataframes:
            st.info("스키마를 탐색하려면 먼저 데이터를 업로드하세요.")
            return
        
        view_option = st.radio(
            "보기 방식 선택:",
            ["테이블 스키마", "관계 다이어그램"],
            horizontal=True
        )
        
        if view_option == "테이블 스키마":
            # 개별 테이블 스키마 표시
            for name, df in st.session_state.active_dataframes.items():
                with st.expander(f"{name} 스키마", expanded=True):
                    schema_info = self._get_schema_info(df)
                    st.table(schema_info)
                    
                    # 공통 키 후보 찾기
                    common_keys = self._find_common_keys(name, df)
                    if common_keys:
                        st.markdown("##### 🔍 잠재적 조인 키:")
                        for other_df, keys in common_keys.items():
                            st.markdown(f"**{other_df}**와(과) 공통: {', '.join(keys)}")
        
        else:  # 관계 다이어그램
            st.subheader("데이터셋 관계 다이어그램")
            
            if len(st.session_state.active_dataframes) < 2:
                st.info("관계 다이어그램을 보려면 최소 2개 이상의 데이터셋이 필요합니다.")
            else:
                # 관계 그래프 생성
                fig = self._create_relationship_diagram()
                st.plotly_chart(fig, use_container_width=True)

    def _get_schema_info(self, df):
        """데이터프레임의 스키마 정보를 추출"""
        schema_data = []
        
        for col in df.columns:
            # 데이터 타입
            dtype = str(df[col].dtype)
            
            # 고유값 수
            unique_count = df[col].nunique()
            
            # 결측치 수와 비율
            null_count = df[col].isnull().sum()
            null_percent = (null_count / len(df) * 100).round(2) if len(df) > 0 else 0
            
            # 샘플 값 (최대 3개)
            sample_values = df[col].dropna().head(3).tolist()
            sample_str = str(sample_values)[:50] + ('...' if len(str(sample_values)) > 50 else '')
            
            schema_data.append({
                '컬럼명': col,
                '데이터타입': dtype,
                '고유값수': unique_count,
                '결측치': f"{null_count} ({null_percent}%)",
                '샘플값': sample_str
            })
            
        return pd.DataFrame(schema_data)

    def _find_common_keys(self, current_df_name, current_df):
        """다른 데이터프레임과의 공통 키 후보를 찾음"""
        common_keys = {}
        
        for other_name, other_df in st.session_state.active_dataframes.items():
            if other_name != current_df_name:
                # 두 데이터프레임 간의 공통 열 찾기
                common_columns = set(current_df.columns) & set(other_df.columns)
                
                # 공통 열 중에서 키 후보 평가
                potential_keys = []
                for col in common_columns:
                    # 두 데이터프레임 모두에서 고유값이 많은 열이 키 후보
                    if current_df[col].nunique() > 0.5 * len(current_df) or \
                       other_df[col].nunique() > 0.5 * len(other_df) or \
                       col.lower().endswith('_id') or 'id' in col.lower():
                        potential_keys.append(col)
                
                if potential_keys:
                    common_keys[other_name] = potential_keys
        
        return common_keys

    def _create_relationship_diagram(self):
        """데이터셋 간의 관계 다이어그램 생성"""
        # 노드(테이블)와 엣지(관계) 정보 생성
        nodes = []
        edges = []
        
        # 각 데이터프레임에 대한 노드 생성
        for i, (name, df) in enumerate(st.session_state.active_dataframes.items()):
            # 노드 추가
            nodes.append({
                'id': name,
                'label': f"{name}<br>({df.shape[0]} rows, {df.shape[1]} cols)",
                'shape': 'box',
                'color': f"hsl({(i * 137) % 360}, 70%, 60%)"  # 고르게 분포된 색상
            })
            
            # 다른 데이터프레임과의 관계 찾기
            for other_name, other_df in list(st.session_state.active_dataframes.items())[i+1:]:
                common_columns = set(df.columns) & set(other_df.columns)
                
                # 공통 열이 있으면 엣지 추가
                for col in common_columns:
                    if col.lower().endswith('_id') or 'id' in col.lower():
                        edges.append({
                            'from': name,
                            'to': other_name,
                            'label': col,
                            'width': 2
                        })
                        break  # 하나의 주요 관계만 표시
                
                # 공통 열은 있지만 ID 컬럼이 없는 경우, 첫 번째 공통 열로 연결
                if common_columns and not any(col.lower().endswith('_id') or 'id' in col.lower() for col in common_columns):
                    edges.append({
                        'from': name,
                        'to': other_name,
                        'label': next(iter(common_columns)),
                        'width': 1,
                        'dashes': True  # 점선으로 표시 (약한 관계)
                    })
        
        # Plotly 그래프로 다이어그램 생성
        G = nx.Graph()
        
        # 노드 추가
        for node in nodes:
            G.add_node(node['id'], **node)
        
        # 엣지 추가
        for edge in edges:
            G.add_edge(edge['from'], edge['to'], **edge)
        
        # 레이아웃 계산 (Spring layout)
        pos = nx.spring_layout(G)
        
        # Plotly 그래프 생성
        edge_traces = []
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                line=dict(width=edge[2].get('width', 1), color='#888', dash='dash' if edge[2].get('dashes', False) else 'solid'),
                hoverinfo='none',
                mode='lines'
            )
            edge_traces.append(edge_trace)
            
            # 엣지 레이블
            if 'label' in edge[2]:
                edge_label_trace = go.Scatter(
                    x=[(x0 + x1) / 2],
                    y=[(y0 + y1) / 2],
                    text=[edge[2]['label']],
                    mode='text',
                    hoverinfo='none',
                    textposition="middle center",
                    textfont=dict(size=10, color='#888')
                )
                edge_traces.append(edge_label_trace)
        
        # 노드 트레이스 생성
        node_trace = go.Scatter(
            x=[pos[node][0] for node in G.nodes()],
            y=[pos[node][1] for node in G.nodes()],
            mode='markers+text',
            text=[G.nodes[node].get('label', node) for node in G.nodes()],
            textposition="middle center",
            hoverinfo='text',
            marker=dict(
                size=30,
                color=[G.nodes[node].get('color', '#1f77b4') for node in G.nodes()],
                line=dict(width=2, color='#ffffff')
            )
        )
        
        # 그래프 생성
        fig = go.Figure(data=edge_traces + [node_trace],
                     layout=go.Layout(
                         showlegend=False,
                         hovermode='closest',
                         margin=dict(b=20, l=5, r=5, t=40),
                         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                         title='데이터셋 관계 다이어그램',
                         plot_bgcolor='#ffffff'
                     ))
        
        return fig

    def _show_merge_tools(self):
        """병합 도구 UI"""
        st.header("데이터 병합 도구")
        
        if len(st.session_state.active_dataframes) < 2:
            st.info("데이터 병합을 위해서는 최소 2개 이상의 데이터셋이 필요합니다.")
            return
            
        # 병합 방식 선택
        merge_method = st.radio(
            "병합 방식 선택:",
            ["시각적 병합 도구", "SQL 쿼리 빌더", "Python 코드"],
            horizontal=True
        )
        
        if merge_method == "시각적 병합 도구":
            self._show_visual_merge_tool()
        elif merge_method == "SQL 쿼리 빌더":
            self._show_sql_query_builder()
        else:
            self._show_python_code_editor()
    
    def _show_visual_merge_tool(self):
        """시각적 병합 도구 UI"""
        st.subheader("시각적 병합 도구")
        
        col1, col2 = st.columns(2)
        
        with col1:
            left_df = st.selectbox(
                "왼쪽 데이터셋 선택:",
                options=list(st.session_state.active_dataframes.keys()),
                key="visual_left_df"
            )

        with col2:
            right_df = st.selectbox(
                "오른쪽 데이터셋 선택:",
                options=[df for df in st.session_state.active_dataframes.keys() if df != left_df],
                key="visual_right_df"
            )
        
        # 두 데이터프레임의 공통 열 찾기
        common_columns = list(set(st.session_state.active_dataframes[left_df].columns) & 
                             set(st.session_state.active_dataframes[right_df].columns))
        
        if not common_columns:
            st.warning("두 데이터셋 간에 공통 열이 없습니다. 직접 연결 열을 지정해주세요.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                left_key = st.selectbox(
                    f"{left_df} 키 열:",
                    options=st.session_state.active_dataframes[left_df].columns,
                    key="visual_left_key"
                )
            
            with col2:
                right_key = st.selectbox(
                    f"{right_df} 키 열:",
                    options=st.session_state.active_dataframes[right_df].columns,
                    key="visual_right_key"
                )
            
            join_columns = [(left_key, right_key)]
        else:
            # 자동 감지된 공통 열 표시 및 선택
            st.success(f"자동으로 {len(common_columns)}개의 공통 열을 감지했습니다.")
            
            # 공통 열 중 조인 키로 사용할 열 선택
            selected_columns = st.multiselect(
                "조인 키로 사용할 열 선택:",
                options=common_columns,
                default=[col for col in common_columns if col.lower().endswith('_id') or 'id' in col.lower()][:1]
            )
            
            if not selected_columns:
                st.warning("조인 키를 하나 이상 선택하세요.")
                return
                
            join_columns = [(col, col) for col in selected_columns]
        
        # 조인 유형 선택
        join_type = st.selectbox(
            "조인 유형:",
            options=["inner", "left", "right", "outer"],
            format_func=lambda x: {
                "inner": "Inner Join (교집합)",
                "left": "Left Join (왼쪽 데이터셋 기준)",
                "right": "Right Join (오른쪽 데이터셋 기준)",
                "outer": "Outer Join (합집합)"
            }[x]
        )
        
        # 열 충돌 처리 방법
        suffix_option = st.radio(
            "열 이름 충돌 처리:",
            options=["접미사 추가", "왼쪽 우선", "오른쪽 우선"],
            horizontal=True
        )
        
        suffixes = ('_left', '_right')
        if suffix_option == "왼쪽 우선":
            # 왼쪽 데이터프레임의 열 이름을 유지하고 오른쪽만 접미사 추가
            suffixes = ('', '_right')
        elif suffix_option == "오른쪽 우선":
            # 오른쪽 데이터프레임의 열 이름을 유지하고 왼쪽만 접미사 추가
            suffixes = ('_left', '')
        
        # 병합 결과 이름
        result_name = st.text_input(
            "병합 결과 이름:",
            value=f"{left_df}_{right_df}_{join_type}"
        )
        
        # 병합 실행 버튼
        if st.button("병합 실행", key="run_visual_merge"):
            try:
                # 시작 시간 측정
                start_time = time.time()
                
                # 데이터프레임 가져오기
                left_data = st.session_state.active_dataframes[left_df]
                right_data = st.session_state.active_dataframes[right_df]
                
                # 멀티 키 조인 처리
                if len(join_columns) == 1:
                    # 단일 키 조인
                    left_key, right_key = join_columns[0]
                    merged_df = pd.merge(
                        left_data,
                        right_data,
                        left_on=left_key,
                        right_on=right_key,
                        how=join_type,
                        suffixes=suffixes
                    )
                else:
                    # 다중 키 조인
                    left_keys = [lk for lk, _ in join_columns]
                    right_keys = [rk for _, rk in join_columns]
                    merged_df = pd.merge(
                        left_data,
                        right_data,
                        left_on=left_keys,
                        right_on=right_keys,
                        how=join_type,
                        suffixes=suffixes
                    )
                
                # 실행 시간 계산
                execution_time = time.time() - start_time
                
                # 결과 저장
                st.session_state.merged_results[result_name] = merged_df
                
                # 병합 기록 저장
                merge_info = {
                    'name': result_name,
                    'left_df': left_df,
                    'right_df': right_df,
                    'join_columns': join_columns,
                    'join_type': join_type,
                    'suffixes': suffixes,
                    'rows': len(merged_df),
                    'columns': len(merged_df.columns),
                    'execution_time': execution_time,
                    'timestamp': time.time()
                }
                st.session_state.merge_history.append(merge_info)
                
                # 성공 메시지 표시
                st.success(f"병합 완료! 결과: {len(merged_df)}행 × {len(merged_df.columns)}열 (실행 시간: {execution_time:.2f}초)")
                
                # 결과 미리보기
                st.subheader("병합 결과 미리보기")
                st.dataframe(merged_df.head(5))
                
                # 병합 결과 활성화 여부
                if st.checkbox("병합 결과를 활성 데이터셋으로 추가", value=True):
                    st.session_state.active_dataframes[result_name] = merged_df
                    st.info(f"'{result_name}'이(가) 활성 데이터셋에 추가되었습니다.")
                
            except Exception as e:
                st.error(f"병합 중 오류 발생: {str(e)}")
                st.session_state.last_error = str(e)
        
        # 고급 옵션 (선택적)
        with st.expander("고급 병합 옵션", expanded=False):
            st.checkbox("중복 행 제거", value=False, key="remove_duplicates")
            st.checkbox("결측값 제거", value=False, key="drop_na")
            st.checkbox("인덱스 재설정", value=True, key="reset_index")
            st.selectbox("타입 불일치 처리:", ["경고만 표시", "자동 변환 시도", "오류 발생"], index=1, key="type_mismatch")

    def _show_sql_query_builder(self):
        """SQL 쿼리 빌더 UI"""
        st.subheader("SQL 쿼리 빌더")
        
        # 데이터프레임 등록 정보
        st.markdown("##### 사용 가능한 테이블:")
        for name in st.session_state.active_dataframes.keys():
            df = st.session_state.active_dataframes[name]
            st.code(f"{name}: {len(df)}행 × {len(df.columns)}열")
        
        # 템플릿 선택
        template_option = st.selectbox(
            "쿼리 템플릿:",
            options=["직접 입력"] + list(self.sql_templates.keys())
        )
        
        # 선택된 템플릿에 따라 초기 쿼리 설정
        initial_query = ""
        if template_option != "직접 입력":
            # 템플릿의 DataFrame 이름을 실제 로드된 이름으로 대체
            template_query = self.sql_templates[template_option]
            df_list = list(st.session_state.active_dataframes.keys())
            
            if "df1" in template_query and len(df_list) > 0:
                template_query = template_query.replace("df1", df_list[0])
            if "df2" in template_query and len(df_list) > 1:
                template_query = template_query.replace("df2", df_list[1])
            if "df3" in template_query and len(df_list) > 2:
                template_query = template_query.replace("df3", df_list[2])
                
            initial_query = template_query
        
        # SQL 에디터
        st.markdown("##### SQL 쿼리 작성:")
        
        # SQL 쿼리 입력 (구문 강조 추가)
        query = st.text_area(
            "SQL 쿼리:",
            value=initial_query,
            height=150,
            help="표준 SQL 구문으로 쿼리를 작성하세요. 로드된 데이터프레임 이름을 테이블 이름으로 사용합니다."
        )
        
        # 서식 지정된 쿼리 표시
        if query:
            with st.expander("서식 지정 쿼리", expanded=True):
                formatted_query = sqlparse.format(query, reindent=True, keyword_case='upper')
                st.code(formatted_query, language="sql")
        
        # 쿼리 실행
        if st.button("쿼리 실행", key="run_sql_query"):
            if not query:
                st.warning("쿼리를 입력하세요.")
                return
                
            try:
                import pandasql as ps
                
                # 시작 시간 측정
                start_time = time.time()
                
                # pandasql에서 사용할 로컬 변수 공간 생성
                local_dict = {name: df for name, df in st.session_state.active_dataframes.items()}
                
                # 쿼리 실행
                result_df = ps.sqldf(query, locals=local_dict)
                
                # 실행 시간 계산
                execution_time = time.time() - start_time
                
                # 결과 이름 생성
                result_name = st.text_input("결과 데이터셋 이름:", value=f"sql_result_{len(st.session_state.merged_results)}")
                
                # 결과 저장
                st.session_state.merged_results[result_name] = result_df
                
                # 병합 기록 저장
                merge_info = {
                    'name': result_name,
                    'type': 'sql',
                    'query': query,
                    'rows': len(result_df),
                    'columns': len(result_df.columns),
                    'execution_time': execution_time,
                    'timestamp': time.time()
                }
                st.session_state.merge_history.append(merge_info)
                
                # 성공 메시지 표시
                st.success(f"쿼리 실행 완료! 결과: {len(result_df)}행 × {len(result_df.columns)}열 (실행 시간: {execution_time:.2f}초)")
                
                # 결과 미리보기
                st.subheader("쿼리 결과 미리보기")
                st.dataframe(result_df.head(5))
                
                # 결과를 활성 데이터셋으로 추가
                if st.checkbox("쿼리 결과를 활성 데이터셋으로 추가", value=True, key="sql_add_active"):
                    st.session_state.active_dataframes[result_name] = result_df
                    st.info(f"'{result_name}'이(가) 활성 데이터셋에 추가되었습니다.")
                
            except Exception as e:
                st.error(f"쿼리 실행 중 오류 발생: {str(e)}")
                st.session_state.last_error = str(e)
                
                # 오류 발생 위치 표시 시도
                error_msg = str(e)
                if "no such column" in error_msg.lower():
                    column_match = re.search(r"no such column: ([^\s]+)", error_msg, re.IGNORECASE)
                    if column_match:
                        column_name = column_match.group(1)
                        st.warning(f"열 '{column_name}'이(가) 존재하지 않습니다. 가용한 열 이름을 확인하세요.")
                        
                        # 비슷한 열 이름 제안
                        all_columns = []
                        for df in st.session_state.active_dataframes.values():
                            all_columns.extend(df.columns.tolist())
                        
                        import difflib
                        similar_columns = difflib.get_close_matches(column_name, all_columns)
                        if similar_columns:
                            st.info(f"혹시 이런 열을 찾으시나요? {', '.join(similar_columns)}")

    def _show_python_code_editor(self):
        """Python 코드 에디터 UI"""
        st.subheader("Python 코드 에디터")
        
        # 데이터프레임 정보 표시
        st.markdown("##### 사용 가능한 데이터프레임:")
        code_info = []
        for name, df in st.session_state.active_dataframes.items():
            code_info.append(f"# {name}: {len(df)}행 × {len(df.columns)}열")
        
        st.code("\n".join(code_info))
        
        # 예제 코드 선택
        example_option = st.selectbox(
            "코드 예제:",
            options=[
                "직접 입력",
                "단순 병합 예제",
                "다중 데이터셋 병합",
                "고급 데이터 변환 후 병합",
                "집계 및 요약 통계"
            ]
        )
        
        # 선택된 예제에 따라 초기 코드 설정
        initial_code = ""
        if example_option == "단순 병합 예제":
            df_names = list(st.session_state.active_dataframes.keys())
            if len(df_names) >= 2:
                initial_code = f"""# 두 데이터프레임 병합
df1 = {df_names[0]}
df2 = {df_names[1]}

# inner join으로 병합
merged_data = pd.merge(
    df1, 
    df2,
    left_on='id',  # 왼쪽 데이터프레임의 조인 키 열
    right_on='id',  # 오른쪽 데이터프레임의 조인 키 열
    how='inner'
)

# 결과 반환 (결과 변수 이름은 반드시 'result'로 지정)
result = merged_data
"""
        elif example_option == "다중 데이터셋 병합":
            df_names = list(st.session_state.active_dataframes.keys())
            if len(df_names) >= 3:
                initial_code = f"""# 여러 데이터프레임 순차적 병합
df1 = {df_names[0]}
df2 = {df_names[1]}
df3 = {df_names[2]}

# 첫 번째 병합
temp = pd.merge(
    df1, 
    df2,
    left_on='id',  # 실제 공통 열 이름으로 대체
    right_on='id',
    how='left'
)

# 두 번째 병합
result = pd.merge(
    temp,
    df3,
    left_on='id',
    right_on='id',
    how='left'
)
"""
        elif example_option == "고급 데이터 변환 후 병합":
            df_names = list(st.session_state.active_dataframes.keys())
            if len(df_names) >= 2:
                initial_code = f"""# 데이터 변환 후 병합
df1 = {df_names[0]}.copy()
df2 = {df_names[1]}.copy()

# 데이터 전처리 및 변환
df1['date'] = pd.to_datetime(df1['date'])  # 날짜 형식 변환
df2['value'] = df2['value'].astype(float)  # 데이터 타입 변환

# 새로운 열 생성
df1['year'] = df1['date'].dt.year
df1['month'] = df1['date'].dt.month

# 필터링
df1 = df1[df1['value'] > 0]
df2 = df2[~df2['category'].isnull()]

# 병합
result = pd.merge(
    df1,
    df2,
    on='id',
    how='inner'
)

# 결과 열 정리
result = result.drop(['temp', 'redundant'], axis=1, errors='ignore')
result = result.rename(columns={'old_name': 'new_name'})
"""
        elif example_option == "집계 및 요약 통계":
            df_names = list(st.session_state.active_dataframes.keys())
            if len(df_names) >= 1:
                initial_code = f"""# 데이터 집계 및 요약
df = {df_names[0]}.copy()

# 날짜 형식 변환 (필요한 경우)
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month

# 그룹별 집계
grouped = df.groupby(['category', 'year']).agg({{
    'value': ['sum', 'mean', 'count'],
    'quantity': ['sum', 'mean']
}})

# 멀티 인덱스 열 이름 정리
grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
grouped = grouped.reset_index()

# 피벗 테이블
if 'category' in df.columns and 'year' in df.columns:
    pivot_table = pd.pivot_table(
        df,
        values='value',
        index='category',
        columns='year',
        aggfunc='sum',
        fill_value=0
    )
    
    result = pivot_table.reset_index()
else:
    result = grouped
"""
        
        # Python 코드 입력
        st.markdown("##### Python 코드 작성:")
        user_code = st.text_area(
            "Python 코드:",
            value=initial_code,
            height=300,
            help="Pandas를 사용하여 데이터 병합 코드를 작성하세요. 최종 결과는 'result' 변수에 저장하세요."
        )
        
        # 코드 실행
        if st.button("코드 실행", key="run_python_code"):
            if not user_code:
                st.warning("코드를 입력하세요.")
                return
                
            try:
                # 시작 시간 측정
                start_time = time.time()
                
                # 코드 실행 환경 설정
                exec_globals = {
                    'pd': pd,
                    'np': np,
                    'result': None,
                    **st.session_state.active_dataframes
                }
                
                # 코드 실행
                exec(user_code, exec_globals)
                
                # 결과 추출
                if 'result' not in exec_globals or exec_globals['result'] is None:
                    st.warning("결과가 'result' 변수에 저장되지 않았습니다.")
                    return
                
                result_df = exec_globals['result']
                
                # DataFrame 타입 검증
                if not isinstance(result_df, pd.DataFrame):
                    st.warning(f"결과가 DataFrame이 아닙니다. 타입: {type(result_df)}")
                    return
                
                # 실행 시간 계산
                execution_time = time.time() - start_time
                
                # 결과 이름 생성
                result_name = st.text_input("결과 데이터셋 이름:", value=f"python_result_{len(st.session_state.merged_results)}")
                
                # 결과 저장
                st.session_state.merged_results[result_name] = result_df
                
                # 병합 기록 저장
                merge_info = {
                    'name': result_name,
                    'type': 'python',
                    'code': user_code,
                    'rows': len(result_df),
                    'columns': len(result_df.columns),
                    'execution_time': execution_time,
                    'timestamp': time.time()
                }
                st.session_state.merge_history.append(merge_info)
                
                # 성공 메시지 표시
                st.success(f"코드 실행 완료! 결과: {len(result_df)}행 × {len(result_df.columns)}열 (실행 시간: {execution_time:.2f}초)")
                
                # 결과 미리보기
                st.subheader("코드 실행 결과 미리보기")
                st.dataframe(result_df.head(5))
                
                # 결과를 활성 데이터셋으로 추가
                if st.checkbox("실행 결과를 활성 데이터셋으로 추가", value=True, key="python_add_active"):
                    st.session_state.active_dataframes[result_name] = result_df
                    st.info(f"'{result_name}'이(가) 활성 데이터셋에 추가되었습니다.")
            
            except Exception as e:
                st.error(f"코드 실행 중 오류 발생: {str(e)}")
                st.session_state.last_error = str(e)
                
                # 오류 라인 표시 시도
                import traceback
                with st.expander("상세 오류 정보", expanded=True):
                    st.code(traceback.format_exc())

    def _show_results_manager(self):
        """결과 관리 UI"""
        st.header("병합 결과 관리")
        
        if not st.session_state.merged_results and not st.session_state.merge_history:
            st.info("아직 병합 결과가 없습니다. 병합 도구를 사용하여 데이터를 병합하세요.")
            return
        
        # 병합 히스토리 표시
        if st.session_state.merge_history:
            st.subheader("병합 기록")
            
            # 병합 기록 테이블 생성
            history_data = []
            for i, hist in enumerate(st.session_state.merge_history):
                history_data.append({
                    '번호': i + 1,
                    '이름': hist['name'],
                    '유형': hist.get('type', 'visual'),
                    '행': hist['rows'],
                    '열': hist['columns'],
                    '실행시간(초)': f"{hist['execution_time']:.2f}",
                    '실행일시': pd.to_datetime(hist['timestamp'], unit='s').strftime('%Y-%m-%d %H:%M:%S')
                })
            
            history_df = pd.DataFrame(history_data)
            st.dataframe(history_df, use_container_width=True)
            
            # 병합 기록 세부정보 보기
            selected_hist_idx = st.selectbox(
                "세부 정보를 볼 기록 선택:",
                options=list(range(1, len(st.session_state.merge_history) + 1)),
                format_func=lambda x: f"{x}. {st.session_state.merge_history[x-1]['name']}"
            )
            
            if selected_hist_idx:
                hist = st.session_state.merge_history[selected_hist_idx - 1]
                with st.expander("병합 기록 세부 정보", expanded=True):
                    if hist.get('type') == 'sql':
                        st.subheader("SQL 쿼리")
                        st.code(hist['query'], language="sql")
                    elif hist.get('type') == 'python':
                        st.subheader("Python 코드")
                        st.code(hist['code'], language="python")
                    else:  # 시각적 병합
                        st.subheader("시각적 병합 정보")
                        st.markdown(f"**왼쪽 데이터셋:** {hist.get('left_df', 'N/A')}")
                        st.markdown(f"**오른쪽 데이터셋:** {hist.get('right_df', 'N/A')}")
                        st.markdown(f"**조인 유형:** {hist.get('join_type', 'N/A')}")
                        join_cols = hist.get('join_columns', [])
                        if join_cols:
                            st.markdown("**조인 열:**")
                            for left, right in join_cols:
                                st.markdown(f"- {left} = {right}")
        
        # 저장된 결과 관리
        if st.session_state.merged_results:
            st.subheader("저장된 병합 결과")
            
            # 결과 목록 선택
            selected_result = st.selectbox(
                "결과 선택:",
                options=list(st.session_state.merged_results.keys())
            )
            
            if selected_result:
                result_df = st.session_state.merged_results[selected_result]
                
                col1, col2, col3 = st.columns([1, 1, 1])
                
                with col1:
                    st.metric("행 수", len(result_df))
                with col2:
                    st.metric("열 수", len(result_df.columns))
                with col3:
                    memory_usage = result_df.memory_usage(deep=True).sum() / (1024 * 1024)
                    st.metric("메모리 사용량", f"{memory_usage:.2f} MB")
                
                # 결과 미리보기
                st.subheader("결과 미리보기")
                st.dataframe(result_df.head(10))
                
                # 결과 내보내기 옵션
                export_format = st.radio(
                    "내보내기 형식:",
                    options=["CSV", "Excel", "JSON"],
                    horizontal=True
                )
                
                if st.button("내보내기", key=f"export_{selected_result}"):
                    try:
                        if export_format == "CSV":
                            csv = result_df.to_csv(index=False)
                            st.download_button(
                                label="CSV 다운로드",
                                data=csv,
                                file_name=f"{selected_result}.csv",
                                mime="text/csv"
                            )
                        elif export_format == "Excel":
                            # 메모리에 Excel 파일 생성
                            from io import BytesIO
                            output = BytesIO()
                            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                result_df.to_excel(writer, index=False)
                            excel_data = output.getvalue()
                            
                            st.download_button(
                                label="Excel 다운로드",
                                data=excel_data,
                                file_name=f"{selected_result}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        else:  # JSON
                            json_str = result_df.to_json(orient="records")
                            st.download_button(
                                label="JSON 다운로드",
                                data=json_str,
                                file_name=f"{selected_result}.json",
                                mime="application/json"
                            )
                    except Exception as e:
                        st.error(f"내보내기 중 오류 발생: {str(e)}")
                
                # 결과 삭제 옵션
                if st.button("결과 삭제", key=f"delete_{selected_result}"):
                    if st.session_state.merged_results.pop(selected_result, None):
                        st.success(f"'{selected_result}' 결과가 삭제되었습니다.")
                        st.rerun()
        
        # 병합 결과 저장 및 로드
        with st.expander("병합 결과 저장 및 로드", expanded=False):
            st.subheader("병합 작업 저장")
            
            preset_name = st.text_input("저장할 작업 이름:", value="Merge_Preset_1")
            
            if st.button("현재 병합 설정 저장"):
                # 현재 설정을 JSON으로 직렬화 가능한 형태로 저장
                merged_results_serializable = {}
                for name, df in st.session_state.merged_results.items():
                    merged_results_serializable[name] = {
                        'columns': df.columns.tolist(),
                        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                        'rows': len(df)
                    }
                
                self.merge_presets[preset_name] = {
                    'active_dataframes': {name: {'shape': list(df.shape)} for name, df in st.session_state.active_dataframes.items()},
                    'merged_results': merged_results_serializable,
                    'merge_history': st.session_state.merge_history,
                    'timestamp': time.time()
                }
                
                st.success(f"병합 작업 '{preset_name}'이(가) 저장되었습니다.")
            
            st.subheader("저장된 작업 목록")
            if self.merge_presets:
                for name, preset in self.merge_presets.items():
                    timestamp = pd.to_datetime(preset['timestamp'], unit='s').strftime('%Y-%m-%d %H:%M:%S')
                    st.markdown(f"**{name}** (저장일시: {timestamp})")
                    st.markdown(f"데이터셋 수: {len(preset['active_dataframes'])}, 결과 수: {len(preset['merged_results'])}")
            else:
                st.info("저장된 병합 작업이 없습니다.")

# 메인 함수
def show():
    """데이터 병합 모듈을 표시하는 메인 함수"""
    merger = DataMerger()
    merger.show()

if __name__ == "__main__":
    show()