import numpy as np
import joblib

# 학습된 모델 로드
model_filename = 'C:/aproject/GRED/0806_svm.joblib'
loaded_model = joblib.load(model_filename)

# 테스트할 RGB 값 입력 (사용자가 테스트할 RGB 값을 여기에 입력)
test_rgb = np.array([[113, 100, 168]])  # 예시 RGB 값 (R, G, B)

# 예측 수행
prediction = loaded_model.predict(test_rgb)

# 예측 결과 출력
print(f"RGB 값 {test_rgb[0]}의 예측 결과: {prediction[0]}")
