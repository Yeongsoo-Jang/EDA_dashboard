# config.py - 설정 및 상수 정의
# 대시보드 설정 및 파라미터를 한 곳에서 관리

# 시각화 색상 테마
COLORSCALES = {
    'correlation': 'RdBu_r',
    'distribution': 'Viridis',
    'categorical': 'Set2'
}

# 분석 파라미터
ANALYSIS_PARAMS = {
    'pca': {
        'max_components': 10,
        'default_components': 2
    },
    'clustering': {
        'max_clusters': 10,
        'default_clusters': 3
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