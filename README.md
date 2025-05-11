# 비즈니스 데이터 분석 대시보드

모듈화된 Streamlit 기반 데이터 분석 대시보드로, CSV 또는 JSON 데이터를 업로드하여 다양한 분석과 시각화를 수행할 수 있습니다.

## 주요 기능

- 📈 **기초 통계 분석**: 데이터의 기본 통계량 및 시각화
- 🔄 **변수 간 상관관계 분석**: 히트맵, 산점도 등을 통한 변수 관계 탐색
- 📊 **변수별 심층 분석**: 분포, 이상치, Q-Q 플롯 등 개별 변수 분석
- 🧠 **고급 EDA**: PCA, 군집화, 3D 시각화, 레이더 차트, 병렬 좌표 그래프
- 🤖 **머신러닝 모델링**: 회귀/분류 모델 자동 구축 및 평가
- 💡 **자동 인사이트 생성**: 데이터에서 주요 패턴과 특이점 자동 발견

## 기술 스택

- **Python 3.8+**
- **Streamlit**: 인터랙티브 웹 인터페이스
- **Pandas & NumPy**: 데이터 처리 및 연산
- **Plotly**: 인터랙티브 시각화
- **Scikit-learn**: 머신러닝 모델링
- **SciPy & StatsModels**: 통계 분석

## 설치 방법

1. 저장소 클론

```bash
git clone https://github.com/Yeongsoo-Jang/EDA_dashboard.git
cd EDA_dashboard
```

2. 가상환경 생성 및 활성화 (선택사항)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

3. 필요 패키지 설치

```bash
pip install -r requirements.txt
```

4. 애플리케이션 실행

```bash
streamlit run app.py
```

## 프로젝트 구조

```
business_dashboard/
├── app.py                  # 메인 애플리케이션 진입점
├── config.py               # 설정 및 상수
├── requirements.txt        # 패키지 의존성
├── README.md               # 프로젝트 설명
├── utils/                  # 유틸리티 함수
│   ├── __init__.py
│   ├── data_loader.py      # 데이터 로딩 함수
│   ├── data_processor.py   # 데이터 전처리 함수
│   └── insights.py         # 인사이트 생성 함수
├── visualizations/         # 시각화 모듈
│   ├── __init__.py
│   ├── basic_viz.py        # 기본 시각화 함수
│   ├── correlation_viz.py  # 상관관계 시각화
│   ├── distribution_viz.py # 분포 시각화
│   └── advanced_viz.py     # 고급 시각화
├── analysis/               # 분석 모듈
│   ├── __init__.py
│   ├── basic_stats.py      # 기초 통계 분석
│   ├── clustering.py       # 군집화 분석
│   ├── pca.py              # PCA 분석
│   ├── time_series.py      # 시계열 분석
│   ├── hypothesis.py       # 가설 검정
│   └── machine_learning.py # 머신러닝 모델링
└── pages/                  # 페이지별 UI 컴포넌트
    ├── __init__.py
    ├── home.py             # 홈페이지
    ├── basic_stats_page.py # 기초 통계 페이지
    ├── variable_page.py    # 변수 분석 페이지
    ├── advanced_page.py    # 고급 분석 페이지
    └── ml_page.py          # 머신러닝 페이지
```

## 사용 방법

1. 애플리케이션 실행 후 웹 브라우저에서 대시보드 접속
2. 왼쪽 사이드바에서 CSV 또는 JSON 파일 업로드 또는 샘플 데이터 선택
3. 사이드바의 '페이지 선택' 라디오 버튼으로 다양한 분석 기능 탐색:
   - **홈**: 데이터 개요 및 요약 정보
   - **기초 통계**: 기본 통계량 및 변수별 분포 시각화
   - **변수 분석**: 개별 변수 상세 분석 및 관계 탐색
   - **고급 EDA**: PCA, 군집화, 고급 시각화 기법
   - **머신러닝 모델링**: 자동 예측 모델 구축 및 평가

## 문제 해결

### 상단 메뉴와 라디오 버튼 메뉴 중복 문제

이 프로젝트는 라디오 버튼 방식의 페이지 전환을 사용합니다. 상단에 나타나는 'advanced_page', 'basic_stats_page' 등의 링크는 사용하지 않습니다. 사이드바의 라디오 버튼만 사용하여 페이지를 전환하세요.

이 문제를 해결하려면 프로젝트 구조를 다음과 같이 변경하세요:

1. `pages` 폴더를 `modules`로 이름 변경
2. `app.py` 파일에서 import 경로 수정:
   ```python
   from modules import home, basic_stats_page, ...
   ```

## 라이센스

MIT License

## 기여하기

1. 이 저장소를 포크합니다
2. 새 기능 브랜치를 생성합니다 (`git checkout -b feature/amazing-feature`)
3. 변경사항을 커밋합니다 (`git commit -m 'Add some amazing feature'`)
4. 브랜치에 푸시합니다 (`git push origin feature/amazing-feature`)
5. Pull Request를 생성합니다