import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


df = pd.read_csv("iris.csv")
print("Tên cột trong file:", df.columns)


# THAY tên cột nếu cần
X = df[['petal_width']].values  
y = df['petal_length'].values  

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện SVR
model = SVR(kernel='linear', C=1.0, epsilon=0.2)
model.fit(X_train, y_train)

# Dự đoán
y_pred = model.predict(X_test)

# Đánh giá
mse = mean_squared_error(y_test, y_pred)
print(f'MSE = {mse:.4f}')

# Vẽ kết quả
sorted_indices = np.argsort(X_test.flatten())
X_test_sorted = X_test[sorted_indices]
y_pred_sorted = y_pred[sorted_indices]

plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test_sorted, y_pred_sorted, color='red', label='Predicted line')
plt.xlabel('Petal_width')
plt.ylabel('Petal_length')
plt.title('Support Vector Regression on Iris dataset')
plt.legend()
plt.grid(True)
plt.show()
