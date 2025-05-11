# analysis/clustering.py - 군집화 분석 관련 함수
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

def perform_kmeans_clustering(df, n_clusters=3):
    """K-means 군집화를 수행합니다."""
    numeric_df = df.select_dtypes(include=['number'])
    
    # 결측치 처리
    numeric_df = numeric_df.fillna(numeric_df.median())
    
    # 데이터 스케일링
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)
    
    # K-means 군집화
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)
    
    # 중심점
    centers = kmeans.cluster_centers_
    
    # 실루엣 점수 계산
    silhouette_avg = silhouette_score(scaled_data, clusters)
    
    # 군집화 결과를 원본 데이터에 추가
    result_df = df.copy()
    result_df['군집'] = clusters
    
    return result_df, centers, silhouette_avg

def perform_dbscan_clustering(df, eps=0.5, min_samples=5):
    """DBSCAN 군집화를 수행합니다."""
    numeric_df = df.select_dtypes(include=['number'])
    
    # 결측치 처리
    numeric_df = numeric_df.fillna(numeric_df.median())
    
    # 데이터 스케일링
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)
    
    # DBSCAN 군집화
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(scaled_data)
    
    # 군집화 결과를 원본 데이터에 추가
    result_df = df.copy()
    result_df['군집'] = clusters
    
    # 군집 수 계산
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    
    return result_df, n_clusters

def get_optimal_clusters(df, max_clusters=10):
    """최적의 군집 수를 찾습니다."""
    numeric_df = df.select_dtypes(include=['number'])
    
    # 결측치 처리
    numeric_df = numeric_df.fillna(numeric_df.median())
    
    # 데이터 스케일링
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)
    
    # 실루엣 점수와 관성 저장을 위한 리스트
    silhouette_scores = []
    inertia_values = []
    
    # 군집 수를 2부터 max_clusters까지 테스트
    for n_clusters in range(2, max_clusters + 1):
        # K-means 군집화
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(scaled_data)
        
        # 실루엣 점수 계산
        silhouette_avg = silhouette_score(scaled_data, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        
        # 관성 저장
        inertia_values.append(kmeans.inertia_)
    
    return list(range(2, max_clusters + 1)), silhouette_scores, inertia_values