import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import joblib  # joblib 라이브러리 임포트

# CSV 파일 로드
csv_path = 'C:/aproject/GRED/output_data_6000_sigma.csv'
df = pd.read_csv(csv_path)

# RGB_within_range가 True인 데이터만 사용
df = df[df['RGB_within_range'] == True]

# Feature와 Label 분리
X = df[['R', 'G', 'B']]
y = df['judge']

# 데이터셋을 학습용과 테스트용으로 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN 모델 학습 (k=4)
knn_model = KNeighborsClassifier(n_neighbors=4)
knn_model.fit(X_train, y_train)

# 모델 저장
model_filename = 'C:/aproject/GRED/0806_knn.joblib'
joblib.dump(knn_model, model_filename)
print(f"모델이 '{model_filename}'로 저장되었습니다.")

# 테스트 데이터에 대한 예측 수행
y_pred = knn_model.predict(X_test)

# 정확도 계산
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

# Classification Report 출력
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# 오차 행렬 계산
conf_matrix = confusion_matrix(y_test, y_pred)

# 오차 행렬 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=knn_model.classes_, yticklabels=knn_model.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('KNN Matrix')
plt.show()

# 3D Plot 생성
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 학습 데이터 산점도
colors = {'danger': 'red', 'warning': 'yellow', 'safe': 'green', 'error': 'blue'}
ax.scatter(X_train['R'], X_train['G'], X_train['B'], c=[colors[label] for label in y_train], marker='o')

# 테스트 데이터에 대한 KNN 모델의 예측 산점도
ax.scatter(X_test['R'], X_test['G'], X_test['B'], c=[colors[label] for label in y_pred], marker='x')

# 축 라벨 설정
ax.set_xlabel('Red')
ax.set_ylabel('Green')
ax.set_zlabel('Blue')

# 플롯 제목 설정
plt.title("3D Plot of RGB Values with KNN Classification")

# 범례 설정
handles = [plt.Line2D([0], [0], marker='o', color='w', label=key, markerfacecolor=val, markersize=10) for key, val in colors.items()]
ax.legend(handles=handles, title='Judge')   

plt.show()
