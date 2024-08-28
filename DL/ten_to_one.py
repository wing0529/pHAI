import os
from PIL import Image
import numpy as np
import pandas as pd

"""
굿
"""

def extract_central_rgb(image_path):
    """이미지의 중앙 픽셀의 RGB 값을 추출"""
    with Image.open(image_path) as img:
        width, height = img.size
        central_pixel = img.getpixel((width // 2, height // 2))
        return central_pixel

def find_closest_image_to_mean(folder_path, output_path):
    """폴더 내에서 평균 RGB 값에 가장 가까운 이미지를 찾아서 저장"""
    images = []
    rgb_values = []
    
    # 이미지 파일 읽기 및 중앙 RGB 값 추출
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(('png', 'jpg', 'jpeg', 'bmp')):
            image_path = os.path.join(folder_path, file_name)
            central_rgb = extract_central_rgb(image_path)
            images.append((file_name, image_path))
            rgb_values.append(central_rgb)
    
    # RGB 값 데이터프레임으로 변환
    df = pd.DataFrame(rgb_values, columns=['R', 'G', 'B'])
    
    # 평균 RGB 값 계산
    mean_values = df.mean()
    
    # 각 이미지의 RGB 값과 평균 간의 유클리드 거리 계산
    distances = df.apply(lambda row: np.linalg.norm(row - mean_values), axis=1)
    
    # 가장 작은 거리의 인덱스 찾기
    closest_index = distances.idxmin()
    
    # 가장 가까운 이미지 선택 및 저장
    closest_image = images[closest_index][1]
    closest_image_name = images[closest_index][0]
    output_image_path = os.path.join(output_path, f"closest_to_mean_{closest_image_name}")
    Image.open(closest_image).save(output_image_path)
    
    print(f"가장 평균에 가까운 이미지는 {closest_image_name}입니다. 저장된 위치: {output_image_path}")

# 폴더 경로와 저장 경로 설정
folder_path = 'C:/aproject/GRED/tenone'  # 이미지가 저장된 폴더 경로
output_path = 'C:/aproject/GRED'  # 결과 이미지를 저장할 폴더 경로

# 함수 실행
find_closest_image_to_mean(folder_path, output_path)
