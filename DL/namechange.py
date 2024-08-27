import os

# 파일이 저장된 디렉터리 경로
folder_path = "./data/first_name"

# pH 값을 3.5에서 6.4까지 0.1씩 증가시키며 리스트 생성
ph_values = [round(x * 0.1, 1) for x in range(35, 65)]

# 파일 리스트를 불러오고, 시간순으로 정렬
files = sorted(os.listdir(folder_path))

file_index = 0

# 각 pH 값에 대해 파일 이름을 생성
for ph in ph_values:
    for trial in range(1, 6):  # 5번의 실험
        for shot in range(1, 11):  # 10장의 사진
            if file_index < len(files):
                old_name = files[file_index]
                new_name = f"{ph}(1)_{trial}_{shot}.jpg"
                old_path = os.path.join(folder_path, old_name)
                new_path = os.path.join(folder_path, new_name)
                os.rename(old_path, new_path)
                file_index += 1
