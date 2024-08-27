import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# CSV 파일 로드
csv_path = 'C:/aproject/GRED/output_data_6000_sigma.csv'
df = pd.read_csv(csv_path)

# RGB_within_range가 True인 데이터만 사용
df = df[df['RGB_within_range'] == True]

# Feature와 Label 분리 (pH 농도를 예측하는 것이 목표)
X = df[['R', 'G', 'B']] 
y = df['pH']  # pH 농도가 목표 변수

# 데이터셋을 학습용과 테스트용으로 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 릿지 회귀 모델 학습
ridge_model = Ridge(alpha=1.0)  # alpha 값은 규제 강도를 조절
ridge_model.fit(X_train, y_train)

# 모델 저장
model_filename = 'C:/aproject/GRED/0806_ridge_model.joblib'
joblib.dump(ridge_model, model_filename)
print(f"모델이 '{model_filename}'로 저장되었습니다.")

# 테스트 데이터에 대한 예측 수행
y_pred = ridge_model.predict(X_test)

# 모델 성능 평가
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Test MSE: {mse:.4f}")
print(f"R^2 Score: {r2:.4f}")

# pH 값을 기준으로 범주화
def categorize_ph(pH_value):
    try:
        pH_float = float(pH_value)
        if 4.8 <= pH_float <= 5.6:
            return 'danger'
        elif 5.7 <= pH_float <= 6.0:
            return 'warning'
        elif 6.1 <= pH_float <= 6.4:
            return 'safe'
        else:
            return 'error'
    except ValueError:
        return 'error'

# 실제 값과 예측 값을 각각 범주화
y_test_categorized = y_test.apply(categorize_ph)
y_pred_categorized = pd.Series(y_pred).apply(categorize_ph)

# 오차 행렬 계산
conf_matrix = confusion_matrix(y_test_categorized, y_pred_categorized)

# 오차 행렬 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test_categorized), yticklabels=np.unique(y_test_categorized))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Ridge Regression Matrix')
plt.show()

# Classification Report 출력
print("\nClassification Report:\n")
print(classification_report(y_test_categorized, y_pred_categorized))

# 예측 결과 출력
print("\nPredicted vs Actual pH Values:")
results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(results_df.head())

# 산점도 플롯 생성
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4, color='red')
plt.xlabel('Actual pH')
plt.ylabel('Predicted pH')
plt.title('Actual vs Predicted pH Values')
plt.show()
