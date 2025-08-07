#dự đoán chiểu rộng cánh hoa dựa vào chiều dài cánh hoa bằng hồi quy tuyến tính

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Đọc dữ liệu
df = pd.read_csv('iris.csv')
X_all = df['petal_length'].values
y_all = df['petal_width'].values

# 2. Tách train/test (80% train, 20% test)
np.random.seed(42)  # để kết quả lặp lại được
indices = np.random.permutation(len(X_all)) # tạo 1 mảng gồm các số từ 0 tới ... nhưng bị xáo trộn
split_idx = int(len(X_all) * 0.8) # tính số lượng mẫu train

train_idx = indices[:split_idx] # lấy ... chỉ số đầu tiên trong mảng
test_idx = indices[split_idx:]

X_train = X_all[train_idx]
y_train = y_all[train_idx]
X_test = X_all[test_idx]
y_test = y_all[test_idx]

# 3. Huấn luyện trên tập train
w = 0.0
b = 0.0
lr = 0.01 # bước nhảy khi cập nhập trọng số và bias trong mỗi vòng lặp
epochs = 100

def compute_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

mse_history = []

for epoch in range(epochs):
    y_pred_train = w * X_train + b
    error = y_train - y_pred_train # sai số giữa thực tế và dự đoán

    dw = -2 * np.mean(X_train * error)
    db = -2 * np.mean(error)

    w -= lr * dw # cập nhật trọng số w theo hướng ngược lại với gradient(giảm loss)
    b -= lr * db

    mse = compute_mse(y_train, y_pred_train)
    mse_history.append(mse) # lưu lại giá trị mse

    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d}: MSE = {mse:.4f}, w = {w:.4f}, b = {b:.4f}")

# 4. Đánh giá trên tập test
y_pred_test = w * X_test + b
mse_test = compute_mse(y_test, y_pred_test)
print(f"\nMSE trên tập test: {mse_test:.4f}")

# 5. Vẽ biểu đồ
plt.figure(figsize=(10, 4))

# (1) Dữ liệu và đường hồi quy trên tập train
plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, label='Train data')
plt.scatter(X_test, y_test, label='Test data', color='orange')
plt.plot(X_all, w * X_all + b, color='red', label='Regression line')
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.title('Linear Regression on Iris')
plt.legend()
plt.grid()

# (2) Quá trình giảm MSE
plt.subplot(1, 2, 2)
plt.plot(mse_history)
plt.xlabel('Epoch')
plt.ylabel('MSE (train)')
plt.title('Training MSE')
plt.grid()

plt.tight_layout()
plt.show()

print(f"\nKết quả mô hình: y = {w:.3f} * x + {b:.3f}")
