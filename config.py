# config.py - 데이터 분석 대시보드 설정 및 상수 정의

# 개별 테마 색상 정의: 오늘의집 기본 팔레트
OHNEUIJIP_DEFAULT_PALETTE = {
    'primary': '#3DBFAD',  # 민트색 (기본)
    'secondary': '#50E3C2', # 밝은 민트
    'tertiary': '#2C8D80',  # 진한 민트
    'accent': '#FF7E36',    # 오렌지 (강조색)
    'background': '#F7F8FA', # 배경
    'text': '#2F3438'       # 텍스트
}

# 애플리케이션에서 사용할 테마들을 정의합니다.
# app.py 에서는 이 BRAND_COLORS 변수를 사용합니다.
BRAND_COLORS = {
    'default': OHNEUIJIP_DEFAULT_PALETTE,  # 기본 테마로 오늘의집 팔레트 사용
    '오늘의집': OHNEUIJIP_DEFAULT_PALETTE, # 명시적으로 "오늘의집" 테마로도 접근 가능
    # 향후 다른 테마 추가 예시 (주석 처리)
    # 'dark_mode': {
    #     'primary': '#77C9D4', # 어두운 테마용 기본색
    #     'secondary': '#57A9B3',
    #     'tertiary': '#3A8A9E',
    #     'accent': '#FFA500', # 어두운 테마용 강조색
    #     'background': '#1E1E1E', # 어두운 배경
    #     'text': '#E0E0E0'       # 밝은 텍스트
    # }
}

# 시각화 색상 테마
COLORSCALES = {
    'correlation': 'RdBu_r',
    'distribution': 'Viridis',
    'categorical': ['#3DBFAD', '#50E3C2', '#2C8D80', '#FF7E36', '#FFA37B', '#FFD0BC'],
    'sequential': 'Mint'
}

# 분석 파라미터
ANALYSIS_PARAMS = {
    'pca': {
        'max_components': 10,
        'default_components': 2
    },
    'clustering': {
        'max_clusters': 10,
        'default_clusters': 4
    },
    'outlier_threshold': 1.5  # IQR 방식의 이상치 감지 임계값
}

# 데이터 유형 분류 기준
DATA_TYPE_MAPPING = {
    'numerical': ['int64', 'float64', 'int32', 'float32'],
    'categorical': ['object', 'category', 'bool'],
    'datetime': ['datetime64[ns]', 'datetime64']
}

# 인사이트 생성을 위한 임계값
INSIGHT_THRESHOLDS = {
    'high_correlation': 0.7,
    'high_missing': 5.0,  # 결측치 백분율
    'high_skew': 1.0,     # 왜도 절대값
    'high_outliers': 5.0  # 이상치 백분율
}

# 오늘의집 제품 카테고리
PRODUCT_CATEGORIES = [
    '가구',
    '패브릭',
    '주방용품',
    '가전',
    '수납/정리',
    '조명',
    '인테리어소품',
    '생활용품',
    '반려동물',
    'DIY/공구',
    '실내식물',
    '가구부속품',
    '서랍/수납장',
    '침구',
    '커튼/블라인드'
]

# 오늘의집 비즈니스 KPI 설정
BUSINESS_KPIS = {
    'revenue': {
        'name': '매출액',
        'unit': '원',
        'format': '{:,.0f}',
        'target_increase': 0.1  # 목표 증가율 (10%)
    },
    'conversion_rate': {
        'name': '전환율',
        'unit': '%',
        'format': '{:.2f}',
        'target_value': 3.0     # 목표값 (3%)
    },
    'average_order_value': {
        'name': '객단가',
        'unit': '원',
        'format': '{:,.0f}',
        'target_value': 50000   # 목표값 (5만원)
    },
    'retention_rate': {
        'name': '재구매율',
        'unit': '%',
        'format': '{:.1f}',
        'target_value': 30.0    # 목표값 (30%)
    },
    'monthly_active_users': {
        'name': 'MAU',
        'unit': '명',
        'format': '{:,.0f}',
        'target_increase': 0.05  # 목표 증가율 (5%)
    }
}

# 오늘의집 사용자 세그먼트 정의
USER_SEGMENTS = {
    'loyal_customers': {
        'name': '충성 고객',
        'description': '6개월 내 3회 이상 구매한 고객'
    },
    'high_value_customers': {
        'name': '고가치 고객',
        'description': '평균 주문금액 상위 20% 고객'
    },
    'new_customers': {
        'name': '신규 고객',
        'description': '최근 30일 내 첫 구매 고객'
    },
    'at_risk_customers': {
        'name': '이탈 위험 고객',
        'description': '90일 이상 비활성 고객'
    },
    'browsers': {
        'name': '브라우저',
        'description': '방문만 하고 구매하지 않는 고객'
    }
}

APP_CONFIG = {
    "large_data_threshold": 500000,  # 행 * 열이 이 값보다 크면 대용량으로 간주
    "default_theme": "default",
    "cache_ttl": 3600  # 캐시 유효 시간(초)
}