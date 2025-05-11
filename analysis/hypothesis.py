# analysis/hypothesis.py - 가설 검정 관련 함수
import pandas as pd
import numpy as np
from scipy import stats

def perform_ttest(df, group_column, value_column, alpha=0.05):
    """두 그룹 간의 t-검정을 수행합니다."""
    # 그룹 확인
    groups = df[group_column].unique()
    
    if len(groups) != 2:
        return None, "t-검정은 정확히 두 개의 그룹이 필요합니다."
    
    # 그룹별 데이터 추출
    group1_data = df[df[group_column] == groups[0]][value_column].dropna()
    group2_data = df[df[group_column] == groups[1]][value_column].dropna()
    
    # t-검정 수행
    t_stat, p_value = stats.ttest_ind(group1_data, group2_data, equal_var=False)
    
    # 결과 해석
    if p_value < alpha:
        conclusion = f"유의수준 {alpha}에서 두 그룹의 평균 차이가 통계적으로 유의합니다."
    else:
        conclusion = f"유의수준 {alpha}에서 두 그룹의 평균 차이가 통계적으로 유의하지 않습니다."
    
    # 결과 정리
    result = {
        '그룹1': groups[0],
        '그룹2': groups[1],
        '그룹1 평균': group1_data.mean(),
        '그룹2 평균': group2_data.mean(),
        '평균 차이': group1_data.mean() - group2_data.mean(),
        't-통계량': t_stat,
        'p-값': p_value,
        '결론': conclusion
    }
    
    return result, None

def perform_anova(df, group_column, value_column, alpha=0.05):
    """분산 분석(ANOVA)을 수행합니다."""
    # 그룹 확인
    groups = df[group_column].unique()
    
    if len(groups) < 3:
        return None, "ANOVA는 세 개 이상의 그룹에 적합합니다. 두 그룹의 경우 t-검정을 고려하세요."
    
    # 그룹별 데이터 추출
    group_data = []
    for group in groups:
        group_data.append(df[df[group_column] == group][value_column].dropna())
    
    # ANOVA 수행
    f_stat, p_value = stats.f_oneway(*group_data)
    
    # 결과 해석
    if p_value < alpha:
        conclusion = f"유의수준 {alpha}에서 그룹 간 평균 차이가 통계적으로 유의합니다."
    else:
        conclusion = f"유의수준 {alpha}에서 그룹 간 평균 차이가 통계적으로 유의하지 않습니다."
    
    # 결과 정리
    result = {
        '그룹 수': len(groups),
        '그룹별 평균': {str(group): data.mean() for group, data in zip(groups, group_data)},
        'F-통계량': f_stat,
        'p-값': p_value,
        '결론': conclusion
    }
    
    return result, None

def perform_chi2_test(df, var1, var2, alpha=0.05):
    """두 범주형 변수 간의 카이제곱 검정을 수행합니다."""
    # 교차표 생성
    contingency_table = pd.crosstab(df[var1], df[var2])
    
    # 카이제곱 검정 수행
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    
    # 결과 해석
    if p_value < alpha:
        conclusion = f"유의수준 {alpha}에서 {var1}와 {var2} 간에 통계적으로 유의한 관계가 있습니다."
    else:
        conclusion = f"유의수준 {alpha}에서 {var1}와 {var2} 간에 통계적으로 유의한 관계가 없습니다."
    
    # 결과 정리
    result = {
        '변수1': var1,
        '변수2': var2,
        '카이제곱 통계량': chi2,
        '자유도': dof,
        'p-값': p_value,
        '결론': conclusion,
        '교차표': contingency_table.to_dict()
    }
    
    return result, None