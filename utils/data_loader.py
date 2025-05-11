# utils/data_loader.py - 데이터 로딩 관련 함수
import pandas as pd
import streamlit as st
from io import StringIO
import json
import numpy as np

@st.cache_data
def load_data(file):
    """파일에서 데이터를 로드하고 기본 전처리를 수행합니다."""
    if file.name.endswith('.csv'):
        data = pd.read_csv(file)
    elif file.name.endswith('.json'):
        data = pd.read_json(file)
    
    # 날짜 형식 자동 변환
    for col in data.columns:
        if data[col].dtype == 'object':
            try:
                data[col] = pd.to_datetime(data[col])
            except:
                pass
    
    return data

def generate_sample_data(sample_type="sales"):
    """샘플 데이터를 생성합니다."""
    if sample_type == "sales":
        # 판매 데이터 샘플 생성 로직
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        
        products = ['제품A', '제품B', '제품C', '제품D']
        regions = ['서울', '부산', '대구', '인천', '광주']
        
        sales_data = []
        
        for _ in range(1000):
            date = np.random.choice(dates)
            product = np.random.choice(products)
            region = np.random.choice(regions)
            quantity = np.random.randint(1, 50)
            price = np.random.choice([15000, 28000, 35000, 42000])
            discount = np.random.choice([0, 0, 0, 0.1, 0.2])
            
            sales_data.append({
                '날짜': date,
                '제품': product,
                '지역': region,
                '수량': quantity,
                '가격': price,
                '할인율': discount,
                '매출액': round(quantity * price * (1 - discount))
            })
        
        sample_df = pd.DataFrame(sales_data)
        
    elif sample_type == "customer":
        # 고객 데이터 샘플 생성 로직
        np.random.seed(42)
        
        customer_data = []
        
        for i in range(500):
            age = np.random.randint(18, 70)
            gender = np.random.choice(['남성', '여성'])
            income = np.random.normal(50000, 15000)
            spending = income * np.random.normal(0.3, 0.1) + 5000
            visits = np.random.poisson(8)
            satisfaction = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.1, 0.2, 0.4, 0.25])
            membership = np.random.choice(['일반', '실버', '골드', 'VIP'], p=[0.4, 0.3, 0.2, 0.1])
            
            customer_data.append({
                '고객ID': f'CUST{i+1001}',
                '나이': age,
                '성별': gender,
                '연소득': round(income),
                '연간지출액': round(spending),
                '방문횟수': visits,
                '만족도': satisfaction,
                '회원등급': membership
            })
        
        sample_df = pd.DataFrame(customer_data)
    
    else:
        # 기본 샘플 데이터
        sample_df = pd.DataFrame({
            'x': np.random.normal(0, 1, 100),
            'y': np.random.normal(0, 1, 100),
            'category': np.random.choice(['A', 'B', 'C'], 100)
        })
    
    # 데이터프레임을 CSV 문자열로 변환
    csv_str = sample_df.to_csv(index=False)
    sample_file = StringIO(csv_str)
    
    if sample_type == "sales":
        sample_file.name = "판매_데이터_샘플.csv"
    elif sample_type == "customer":
        sample_file.name = "고객_데이터_샘플.csv"
    else:
        sample_file.name = "기본_샘플_데이터.csv"
    
    return sample_file