import os
import re

def count_images_in_directory(directory_path):
    # 이미지 확장자 목록
    image_extensions = [r'\.jpg', r'\.jpeg', r'\.png', r'\.gif', r'\.bmp', r'\.tiff', r'\.svg', r'\.webp']

    image_count = 0

    # 디렉터리 내의 모든 파일 탐색
    for root, dirs, files in os.walk(directory_path):
        for file_name in files:
            # 파일 이름이 이미지 확장자로 끝나는지 확인
            if any(re.search(ext, file_name, re.IGNORECASE) for ext in image_extensions):
                image_count += 1

    return image_count

# 디렉터리 경로 설정
directory_path = 'C:/aproject/pHAI/data/first'

# 이미지 개수 세기
image_count = count_images_in_directory(directory_path)
print(f"디렉터리 내 이미지 개수: {image_count}개")
