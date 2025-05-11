# analysis/pca.py - PCA 분석 관련 함수
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def perform_pca(df, n_components=2):
    """주성분 분석(PCA)을 수행합니다."""
    numeric_df = df.select_dtypes(include=['number'])
    
    # 결측치 처리
    numeric_df = numeric_df.fillna(numeric_df.median())
    
    # 데이터 스케일링
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)
    
    # PCA 수행
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(scaled_data)
    
    # PCA 결과를 데이터프레임으로 변환
    columns = [f'PC{i+1}' for i in range(n_components)]
    pca_df = pd.DataFrame(data=pca_result, columns=columns)
    
    # 원본 데이터프레임과 병합
    categorical_cols = df.select_dtypes(exclude=['number']).columns
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            pca_df[col] = df[col].values
    
    # 설명된 분산 비율
    explained_variance = pca.explained_variance_ratio_
    
    # 각 주성분에 대한 변수 기여도 (로딩)
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=columns,
        index=numeric_df.columns
    )
    
    return pca_df, explained_variance, loadings

def get_pca_biplot_data(df, n_components=2):
    """PCA 바이플롯을 위한 데이터를 생성합니다."""
    pca_df, explained_variance, loadings = perform_pca(df, n_components)
    
    # 스케일링 팩터 (로딩을 스케일링하여 동일한 플롯에 표시)
    scaling = np.max(np.abs(pca_df.iloc[:, 0:n_components].values)) / np.max(np.abs(loadings.values)) * 0.7
    
    # 스케일링된 로딩 값
    scaled_loadings = loadings * scaling
    
    return pca_df, explained_variance, scaled_loadings