import numpy as np
import joblib

# 모델 로드
model_filename = 'C:/aproject/GRED/0806_ridge_model.joblib'
ridge_model = joblib.load(model_filename)

# pH 값을 기준으로 범주화하는 함수
def categorize_ph(pH_value):
    if 4.8 <= pH_value <= 5.6:
        return 'danger'
    elif 5.7 <= pH_value <= 6.0:
        return 'warning'
    elif 6.1 <= pH_value <= 6.4:
        return 'safe'
    else:
        return 'error'

# RGB 입력으로 pH 값을 예측하는 함수
def predict_pH(R, G, B):
    input_features = np.array([[R, G, B]])  # 입력 데이터를 배열로 변환
    predicted_pH = ridge_model.predict(input_features)[0]  # 예측 실행
    pH_category = categorize_ph(predicted_pH)  # 예측된 pH 값을 범주화
    return predicted_pH, pH_category

# 사용 예
R, G, B = 113, 100, 168  # RGB 값을 직접 입력
predicted_pH, pH_category = predict_pH(R, G, B)
print(f"입력 RGB: ({R}, {G}, {B}) -> 예측 pH: {predicted_pH:.2f}, 카테고리: {pH_category}")
