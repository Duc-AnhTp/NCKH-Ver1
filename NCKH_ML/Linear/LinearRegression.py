#dự đoán chiểu rộng cánh hoa dựa vào chiều dài cánh hoa bằng hồi quy tuyến tính

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('iris.csv')

# chọn 1 cột cần dùng cho hổi quy
X = df['petal_length'].values
y = df['petal_width'].values

#khởi tạo trọng số và siêu tham số
w = 0.0
b = 0.0
lr = 0.01
epochs = 100 # số vòng lặp.

def compute_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

mse_history = []

for epoch in range(epochs):
    y_pred = w * X + b
    error = y - y_pred 

    dw = -2 * np.mean(X * error)
    db = -2 * np.mean(error)

    w -= lr * dw
    b -= lr * db

    mse = compute_mse(y, y_pred)
    mse_history.append(mse)

    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d}: MSE = {mse: .4f}, w = {w:.4f}, b = {b:.4f}")

# ===== 6. Vẽ biểu đồ kết quả =====
plt.figure(figsize=(10, 4))

# (1) Vẽ dữ liệu và đường hồi quy
plt.subplot(1, 2, 1)
plt.scatter(X, y, label='Dữ liệu thực')
plt.plot(X, w * X + b, color='red', label='Đường hồi quy')
plt.xlabel('Petal length (cm)')
plt.ylabel('Petal width (cm)')
plt.title('Linear Regression trên dữ liệu Iris')
plt.legend()
plt.grid()

# (2) Vẽ quá trình giảm MSE
plt.subplot(1, 2, 2)
plt.plot(mse_history)
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('MSE theo thời gian')
plt.grid()

plt.tight_layout()
plt.show()


print(f"\n kết quả mô hình: y = {w:.3f} * x + {b:.3f}")
