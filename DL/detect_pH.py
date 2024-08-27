import os
from collections import defaultdict

# 폴더 경로
folder_path = 'C:/aproject/pHAI/data/first'

# pH 농도를 저장할 딕셔너리 (기본값 0)
pH_counts = defaultdict(int)

# 폴더 내 파일을 순회
for filename in os.listdir(folder_path):
    if filename.endswith('.jpg'):
        # 파일명에서 pH 값 추출 (예: '3.5(1)_~.jpg' 에서 '3.5' 추출)
        try:
            pH_value = filename.split('(')[0]  # '(1)' 앞의 숫자 추출
            pH_float = float(pH_value)  # 숫자로 변환

            # pH 값이 3.5부터 6.4 사이인지 확인
            if 3.5 <= pH_float <= 6.4:
                pH_counts[pH_float] += 1  # 해당 pH 농도의 사진 수 증가

        except ValueError:
            # 숫자로 변환할 수 없는 경우(파일명 형식 오류 등)는 무시
            continue

# 결과 출력
for pH_value in sorted(pH_counts.keys()):
    print(f"pH {pH_value}: {pH_counts[pH_value]}장")
