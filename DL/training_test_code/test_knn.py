import joblib
import numpy as np

# 저장된 모델 로드
model_filename = 'C:/aproject/GRED/0806_knn.joblib'
knn_model = joblib.load(model_filename)

# 새로운 RGB 값 입력 (예: R=128, G=64, B=192)
new_rgb = np.array([[113, 100, 168]])

# 예측 수행
prediction = knn_model.predict(new_rgb)

# 예측 결과 출력
print(f"Predicted class for RGB({new_rgb[0][0]}, {new_rgb[0][1]}, {new_rgb[0][2]}) is: {prediction[0]}")
