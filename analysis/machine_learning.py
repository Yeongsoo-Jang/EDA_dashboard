# analysis/machine_learning.py - 머신러닝 모델링 관련 함수
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report

def prepare_ml_data(df, target_column, test_size=0.2, random_state=42):
    """머신러닝을 위한 데이터를 준비합니다."""
    # 목표 변수가 존재하는지 확인
    if target_column not in df.columns:
        return None, None, None, None, "목표 변수가 존재하지 않습니다."
    
    # 목표 변수 유형 확인
    if pd.api.types.is_numeric_dtype(df[target_column]):
        problem_type = 'regression'
    else:
        problem_type = 'classification'
    
    # 독립 변수와 종속 변수 분리
    y = df[target_column]
    X = df.drop(columns=[target_column])
    
    # 범주형 변수 제외 (원-핫 인코딩 필요)
    X = X.select_dtypes(include=['number'])
    
    # 결측치 제거
    X = X.fillna(X.median())
    
    # 훈련 세트와 테스트 세트 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test, problem_type

def train_regression_model(X_train, X_test, y_train, y_test):
    """회귀 모델을 훈련하고 평가합니다."""
    # 선형 회귀 모델
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
    lr_r2 = r2_score(y_test, lr_pred)
    
    # 랜덤 포레스트 회귀 모델
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    rf_r2 = r2_score(y_test, rf_pred)
    
    # 특성 중요도 계산
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # 결과 정리
    results = {
        'linear_regression': {
            'model': lr_model,
            'rmse': lr_rmse,
            'r2': lr_r2
        },
        'random_forest': {
            'model': rf_model,
            'rmse': rf_rmse,
            'r2': rf_r2
        },
        'feature_importance': feature_importance
    }
    
    return results

def train_classification_model(X_train, X_test, y_train, y_test):
    """분류 모델을 훈련하고 평가합니다."""
    # 로지스틱 회귀 모델
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_accuracy = accuracy_score(y_test, lr_pred)
    lr_report = classification_report(y_test, lr_pred, output_dict=True)
    
    # 랜덤 포레스트 분류 모델
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    rf_report = classification_report(y_test, rf_pred, output_dict=True)
    
    # 특성 중요도 계산
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # 결과 정리
    results = {
        'logistic_regression': {
            'model': lr_model,
            'accuracy': lr_accuracy,
            'classification_report': lr_report
        },
        'random_forest': {
            'model': rf_model,
            'accuracy': rf_accuracy,
            'classification_report': rf_report
        },
        'feature_importance': feature_importance
    }
    
    return results