# utils/data_loader.py - 오늘의집 데이터 로딩 관련 함수
import pandas as pd
import streamlit as st
from io import StringIO
import numpy as np
from datetime import datetime, timedelta, timezone
import json
import os
import time
from config import PRODUCT_CATEGORIES, USER_SEGMENTS

@st.cache_data(ttl=3600, show_spinner=False)
def load_data(file):
    """파일에서 데이터를 로드하고 기본 전처리를 수행합니다."""
    try:
        file_extension = os.path.splitext(file.name)[1].lower()
        
        # 파일 크기 확인 (대용량 파일 처리)
        file_size_mb = file.size / (1024 * 1024)
        if file_size_mb > 100:  # 100MB 이상인 경우
            st.warning(f"파일 크기가 {file_size_mb:.1f}MB로 큽니다. 로딩 시간이 오래 걸릴 수 있습니다.")
        
        start_time = time.time()
        
        if file_extension == '.csv':
            # CSV 파일 인코딩 자동 감지 시도
            try:
                # 처음에는 UTF-8 시도
                data = pd.read_csv(file, low_memory=False)
            except UnicodeDecodeError:
                # UTF-8 실패 시 CP949(한국어 Windows) 시도
                file.seek(0)  # 파일 포인터 다시 처음으로
                data = pd.read_csv(file, encoding='cp949', low_memory=False)
                
        elif file_extension == '.json':
            data = pd.read_json(file)
        elif file_extension in ['.xlsx', '.xls']:
            # 엑셀 파일의 모든 시트 가져오기
            xls = pd.ExcelFile(file)
            sheet_names = xls.sheet_names
            
            if len(sheet_names) > 1:
                selected_sheet = st.sidebar.selectbox(
                    "사용할 시트를 선택하세요:",
                    options=sheet_names
                )
                data = pd.read_excel(file, sheet_name=selected_sheet)
            else:
                data = pd.read_excel(file)
        else:
            st.error(f"지원하지 않는 파일 형식입니다: {file_extension}")
            return None
        
        # 데이터 로딩 시간 기록
        loading_time = time.time() - start_time
        st.success(f"데이터 로딩 완료! ({loading_time:.2f}초)")
        
        # 자동 전처리
        data, redundant_cols_info_from_preprocessing = preprocess_data(data)
        
        return data, redundant_cols_info_from_preprocessing
    
    except Exception as e:
        st.error(f"데이터 로딩 중 오류 발생: {str(e)}")
        return None, [] # Return None for data and empty list for info on error

def preprocess_data(data):
    """데이터를 자동으로 전처리합니다."""
    # 열 이름 공백 제거 및 소문자 변환
    data.columns = [col.strip().lower().replace(' ', '_') for col in data.columns]
    
    # 날짜 형식 자동 변환
    for col in data.columns:
        if data[col].dtype == 'object':
            # 날짜 형식 변환 시도
            if any(date_keyword in col.lower() for date_keyword in ['date', 'time', '날짜', '일자', '시간']):
                try:
                    data[col] = pd.to_datetime(data[col], errors='coerce')
                except:
                    pass
            
            # 금액 문자열 변환 시도
            if any(price_keyword in col.lower() for price_keyword in ['price', 'amount', 'cost', 'revenue', '금액', '가격']):
                try:
                    # 통화 기호, 쉼표 등 제거
                    data[col] = data[col].astype(str).str.replace(r'[₩,\$,￦,¥,€,£,\s]', '', regex=True)
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                except:
                    pass

    # 불필요한 열 식별 (모든 값이 동일하거나 거의 모든 값이 NA인 경우)
    redundant_cols_info = [] 
    for col in data.columns:
        # 모든 값이 동일한 경우
        if data[col].nunique() == 1:
            redundant_cols_info.append({
                "name": col, 
                "reason": "all_same", 
                "value": data[col].iloc[0] if len(data[col]) > 0 else "N/A"
            })
        # 90% 이상이 NA인 경우
        elif data[col].isna().mean() > 0.9:
            redundant_cols_info.append({
                "name": col, 
                "reason": "mostly_na", 
                "percentage_na": data[col].isna().mean() * 100
            })
    
    # The UI for displaying and acting on redundant_cols_info
    # will be handled in app.py, outside the st.status block.
    # This function no longer creates Streamlit UI elements directly
    # and does not drop columns based on UI interaction within it.
    
    return data, redundant_cols_info

def generate_sample_product_data():
    """오늘의집 제품 데이터 샘플을 생성합니다."""
    np.random.seed(42)
    
    # 현재 날짜를 기준으로 과거 데이터 생성
    now = datetime.now()
    end_date = now.strftime('%Y-%m-%d')
    start_date = (now - timedelta(days=365)).strftime('%Y-%m-%d')
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # 제품 정의
    products = []
    product_ids = range(1001, 1101)  # 100개 제품
    
    for product_id in product_ids:
        category = np.random.choice(PRODUCT_CATEGORIES)
        
        # 카테고리별로 가격 범위 다르게 설정
        if category in ['가구', '가전']:
            base_price = np.random.randint(100000, 1000000)
        elif category in ['패브릭', '조명', '수납/정리']:
            base_price = np.random.randint(30000, 150000)
        else:
            base_price = np.random.randint(5000, 50000)
            
        product_name = f"{category} 제품 {product_id}"
        brand = np.random.choice(['오늘의집', '이케아', '한샘', '까사미아', '리바트', '데코뷰', '무인양품', '니토리', '루이스폴센', '시디즈'], p=[0.3, 0.15, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05])
        
        # 별점과 리뷰 수
        rating = round(max(1.0, min(5.0, np.random.normal(4.2, 0.5))), 1)
        review_count = int(np.random.exponential(scale=50)) + 1
        
        # 배송 유형
        delivery_type = np.random.choice(['일반배송', '특급배송', '무료배송', '설치배송'], p=[0.4, 0.2, 0.3, 0.1])
        
        # 재고 상태
        stock_status = np.random.choice(['재고있음', '재고부족', '품절'], p=[0.7, 0.2, 0.1])
        
        products.append({
            'product_id': product_id,
            'product_name': product_name,
            'category': category,
            'brand': brand,
            'price': base_price,
            'discount_rate': np.random.choice([0, 0, 0, 5, 10, 15, 20, 30]) / 100,
            'rating': rating,
            'review_count': review_count,
            'delivery_type': delivery_type,
            'stock_status': stock_status,
            'is_best': np.random.choice([True, False], p=[0.2, 0.8]),
            'upload_date': np.random.choice(dates[:-90])  # 제품은 최소 90일 전에 업로드
        })
    
    products_df = pd.DataFrame(products)
    
    return products_df

def generate_sample_order_data(products_df, n_orders=2000):
    """오늘의집 주문 데이터 샘플을 생성합니다."""
    np.random.seed(43)
    
    # 현재 날짜를 기준으로 과거 데이터 생성
    now = datetime.now()
    end_date = now.strftime('%Y-%m-%d')
    start_date = (now - timedelta(days=365)).strftime('%Y-%m-%d')
    dates = pd.date_range(start=start_date, end=end_date, freq='H')
    
    # 사용자 ID 생성
    user_ids = range(1, 501)  # 500명의 사용자
    
    # 주문 생성
    orders = []
    for i in range(n_orders):
        order_id = f"ORDER-{i+10001}"
        user_id = np.random.choice(user_ids)
        order_date = np.random.choice(dates)
        
        # 계절성 - 특정 카테고리는 특정 시즌에 인기
        order_date_pd = pd.Timestamp(order_date)
        month = order_date_pd.month
        if month in [12, 1, 2]:  # 겨울
            seasonal_categories = ['패브릭', '침구', '커튼/블라인드', '가전']
        elif month in [3, 4, 5]:  # 봄
            seasonal_categories = ['인테리어소품', '실내식물', '커튼/블라인드', '패브릭']
        elif month in [6, 7, 8]:  # 여름
            seasonal_categories = ['가전', '생활용품', '반려동물', '주방용품']
        else:  # 가을
            seasonal_categories = ['인테리어소품', '조명', '가구', '수납/정리']
            
        # 계절성 반영하여 제품 선택 확률 조정
        seasonal_products = products_df[products_df['category'].isin(seasonal_categories)]
        non_seasonal_products = products_df[~products_df['category'].isin(seasonal_categories)]
        
        if len(seasonal_products) > 0 and np.random.random() < 0.6:  # 60% 확률로 시즌 제품 선택
            selected_product = seasonal_products.sample(1).iloc[0]
        else:
            selected_product = products_df.sample(1).iloc[0]
        
        # 주문 수량
        quantity = np.random.choice([1, 1, 1, 2, 2, 3], p=[0.6, 0.15, 0.1, 0.1, 0.03, 0.02])
        
        # 쿠폰 사용
        coupon_applied = np.random.choice([True, False], p=[0.3, 0.7])
        coupon_amount = round(selected_product['price'] * np.random.choice([0.05, 0.1, 0.15, 0.2])) if coupon_applied else 0
        
        # 결제 수단
        payment_method = np.random.choice(['신용카드', '체크카드', '무통장입금', '네이버페이', '카카오페이', '휴대폰결제'], 
                                        p=[0.4, 0.2, 0.1, 0.15, 0.1, 0.05])
        
        # 배송지역
        region = np.random.choice(['서울', '경기', '인천', '대전', '광주', '대구', '부산', '울산', '세종', '강원', '충북', '충남', '전북', '전남', '경북', '경남', '제주'],
                                p=[0.3, 0.25, 0.05, 0.04, 0.04, 0.05, 0.06, 0.03, 0.01, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.03, 0.02])
        
        # 할인 후 가격 계산
        discounted_price = selected_product['price'] * (1 - selected_product['discount_rate'])
        total_price = discounted_price * quantity - coupon_amount
        
        # 주문 상태
        order_datetime = pd.Timestamp.fromtimestamp(order_date, tz='UTC').replace(tzinfo=None) if isinstance(order_date, (int, float)) else pd.Timestamp(order_date)
        current_time_naive = now.replace(tzinfo=None) if now.tzinfo else now
        days_since_order = (current_time_naive - order_datetime.replace(tzinfo=None)).days

        if days_since_order < 2:
            status = np.random.choice(['결제완료', '상품준비중', '배송중'], p=[0.3, 0.4, 0.3])
        elif days_since_order < 7:
            status = np.random.choice(['배송중', '배송완료', '구매확정'], p=[0.1, 0.3, 0.6])
        else:
            status = '구매확정'
            
        # 리뷰 작성 여부
        review_written = False
        if status == '구매확정' and np.random.random() < 0.6:  # 60%의 확률로 리뷰 작성
            review_written = True
        
        orders.append({
            'order_id': order_id,
            'user_id': user_id,
            'product_id': selected_product['product_id'],
            'product_name': selected_product['product_name'],
            'category': selected_product['category'],
            'brand': selected_product['brand'],
            'order_date': order_date,
            'quantity': quantity,
            'unit_price': selected_product['price'],
            'discount_rate': selected_product['discount_rate'],
            'coupon_applied': coupon_applied,
            'coupon_amount': coupon_amount,
            'total_price': total_price,
            'payment_method': payment_method,
            'region': region,
            'status': status,
            'review_written': review_written
        })
    
    orders_df = pd.DataFrame(orders)
    orders_df['order_date'] = pd.to_datetime(orders_df['order_date'])
    
    return orders_df

def generate_sample_user_data(orders_df):
    """오늘의집 사용자 데이터 샘플을 생성합니다."""
    np.random.seed(44)
    
    # 현재 날짜 기준
    now = datetime.now()
    
    # 주문 데이터에서 고유 사용자 ID 추출
    user_ids = orders_df['user_id'].unique()
    
    # 사용자별 주문 통계 계산
    user_order_stats = {}
    for user_id in user_ids:
        user_orders = orders_df[orders_df['user_id'] == user_id]
        user_order_stats[user_id] = {
            'order_count': len(user_orders),
            'total_spent': user_orders['total_price'].sum(),
            'avg_order_value': user_orders['total_price'].mean(),
            'first_order_date': user_orders['order_date'].min(),
            'last_order_date': user_orders['order_date'].max(),
            'categories_purchased': user_orders['category'].unique(),
            'review_count': user_orders['review_written'].sum()
        }
    
    # 사용자 데이터 생성
    users = []
    for user_id in user_ids:
        stats = user_order_stats[user_id]
        
        # 가입일은 첫 주문일 이전
        days_before_first_order = np.random.randint(1, 60)
        join_date = stats['first_order_date'] - timedelta(days=days_before_first_order)
        
        # 성별
        gender = np.random.choice(['여성', '남성', '미지정'], p=[0.65, 0.3, 0.05])
        
        # 연령대
        age_group = np.random.choice(['20대', '30대', '40대', '50대', '60대 이상'], 
                                    p=[0.25, 0.35, 0.25, 0.1, 0.05])
        
        # 지역은 주문에서 가장 많이 사용된 지역
        if len(orders_df[orders_df['user_id'] == user_id]) > 0:
            region = orders_df[orders_df['user_id'] == user_id]['region'].mode().iloc[0]
        else:
            region = np.random.choice(['서울', '경기', '인천', '기타'], p=[0.4, 0.3, 0.1, 0.2])
        
        # 앱 사용 디바이스
        device = np.random.choice(['iOS', 'Android', '웹'], p=[0.4, 0.45, 0.15])
        
        # 마케팅 이메일 수신 동의
        marketing_opt_in = np.random.choice([True, False], p=[0.7, 0.3])
        
        # 최근 90일 내 활동 여부
        days_since_last_order = (now - stats['last_order_date']).days
        is_active = days_since_last_order <= 90
        
        # 관심 카테고리 (구매 이력 + 랜덤)
        purchased_categories = list(stats['categories_purchased'])
        all_categories = PRODUCT_CATEGORIES.copy()
        for cat in purchased_categories:
            if cat in all_categories:
                all_categories.remove(cat)
        
        interest_count = min(5, len(purchased_categories) + np.random.randint(0, 3))
        interest_categories = purchased_categories.copy()
        while len(interest_categories) < interest_count and all_categories:
            interest_categories.append(np.random.choice(all_categories))
            all_categories.remove(interest_categories[-1])
        
        # 사용자 세그먼트 결정
        if stats['order_count'] >= 3 and days_since_last_order <= 180:
            segment = 'loyal_customers'
        elif stats['avg_order_value'] > 100000:  # 10만원 이상 구매 고객
            segment = 'high_value_customers'
        elif (now - stats['first_order_date']).days <= 30:
            segment = 'new_customers'
        elif days_since_last_order > 90:
            segment = 'at_risk_customers'
        else:
            segment = np.random.choice(['loyal_customers', 'high_value_customers', 'new_customers', 'at_risk_customers', 'browsers'],
                                       p=[0.2, 0.2, 0.2, 0.2, 0.2])
        
        # 최근 30일 방문 횟수 (랜덤)
        recent_visits = 0
        if is_active:
            if segment == 'loyal_customers':
                recent_visits = np.random.randint(5, 30)
            elif segment == 'high_value_customers':
                recent_visits = np.random.randint(3, 15)
            elif segment == 'new_customers':
                recent_visits = np.random.randint(1, 10)
            elif segment == 'at_risk_customers':
                recent_visits = np.random.randint(0, 3)
            else:
                recent_visits = np.random.randint(0, 5)
        
        users.append({
            'user_id': user_id,
            'join_date': join_date,
            'gender': gender,
            'age_group': age_group,
            'region': region,
            'device': device,
            'marketing_opt_in': marketing_opt_in,
            'order_count': stats['order_count'],
            'total_spent': stats['total_spent'],
            'avg_order_value': stats['avg_order_value'],
            'first_order_date': stats['first_order_date'],
            'last_order_date': stats['last_order_date'],
            'days_since_last_order': days_since_last_order,
            'is_active': is_active,
            'interest_categories': ', '.join(interest_categories),
            'review_count': stats['review_count'],
            'recent_visits': recent_visits,
            'user_segment': segment
        })
    
    users_df = pd.DataFrame(users)
    
    # 날짜 열을 datetime 형식으로 변환
    for date_col in ['join_date', 'first_order_date', 'last_order_date']:
        users_df[date_col] = pd.to_datetime(users_df[date_col])
    
    return users_df

def generate_sample_data():
    """샘플 데이터셋을 생성합니다."""
    # 제품 데이터 생성
    products_df = generate_sample_product_data()
    
    # 주문 데이터 생성
    orders_df = generate_sample_order_data(products_df)
    
    # 사용자 데이터 생성
    users_df = generate_sample_user_data(orders_df)
    
    # 데이터를 세션 상태에 저장
    st.session_state['products_df'] = products_df
    st.session_state['orders_df'] = orders_df
    st.session_state['users_df'] = users_df
    
    # 합쳐진 데이터 생성 (주요 분석용)
    # 주문 + 제품 + 사용자 정보를 결합
    merged_df = orders_df.copy()
    
    # 제품 정보 추가
    product_info = products_df[['product_id', 'rating', 'review_count', 'is_best', 'upload_date']].copy()
    merged_df = merged_df.merge(product_info, on='product_id', how='left')
    
    # 사용자 정보 추가
    user_info = users_df[['user_id', 'gender', 'age_group', 'join_date', 'user_segment']].copy()
    merged_df = merged_df.merge(user_info, on='user_id', how='left')
    
    # # CSV 문자열로 변환
    # csv_str = merged_df.to_csv(index=False)
    # sample_file = StringIO(csv_str)
    # sample_file.name = "통합데이터_샘플.csv"
    
    return merged_df, "통합데이터_샘플.csv"