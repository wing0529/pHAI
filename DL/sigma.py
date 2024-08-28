import pandas as pd
import numpy as np

# CSV 파일 로드
df = pd.read_csv('output_dataset.csv')

# pH 농도별로 그룹화하여 RGB 값의 평균과 표준편차 계산
grouped = df.groupby('pH').agg({
    'R': ['mean', 'std'],
    'G': ['mean', 'std'],
    'B': ['mean', 'std']
}).reset_index()

# 컬럼 이름 정리
grouped.columns = ['pH', 'R_mean', 'R_std', 'G_mean', 'G_std', 'B_mean', 'B_std']

# 원본 데이터에 평균과 표준편차 병합
df = pd.merge(df, grouped, on='pH', suffixes=('', '_drop'))

# 병합 후 불필요한 중복 컬럼 제거 (혹시 원본 컬럼이 남아있는 경우 대비)
df = df[[col for col in df.columns if not col.endswith('_drop')]]

# 오차범위 내에 있는지 확인하는 컬럼 추가
df['R_within_range'] = np.abs(df['R'] - df['R_mean']) <= 2 * df['R_std']
df['G_within_range'] = np.abs(df['G'] - df['G_mean']) <= 2 * df['G_std']
df['B_within_range'] = np.abs(df['B'] - df['B_mean']) <= 2 * df['B_std']

# RGB 모두 오차범위 내에 있는지 확인
df['RGB_within_range'] = df['R_within_range'] & df['G_within_range'] & df['B_within_range']

# 결과 출력
within_range_count = df['RGB_within_range'].sum()
total_count = len(df)

print(f"전체 {total_count}개의 데이터 중 {within_range_count}개가 RGB 값이 오차범위 내에 있습니다.")

# 범위를 벗어난 데이터를 확인하려면 아래 코드를 사용할 수 있습니다.
out_of_range = df[~df['RGB_within_range']]
print(out_of_range[['pH', 'R', 'G', 'B', 'R_mean', 'G_mean', 'B_mean', 'R_std', 'G_std', 'B_std']])

# 필요한 컬럼만 선택
df_filtered = df[['judge', 'pH', 'R', 'G', 'B', 'RGB_within_range']]

# 새로운 CSV 파일로 저장
filtered_csv_filename = 'output_dataset_fix.csv'
df_filtered.to_csv(filtered_csv_filename, index=False)

print(f"필터링된 CSV 파일 '{filtered_csv_filename}'이(가) 생성되었습니다.")
