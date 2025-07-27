# dự đoán chiều rộng cánh hoa dựa trên chiều dài cánh hoa và chiều rộng đài hoa, chiều dài đài hoa.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Đọc dữ liệu
df = pd.read_csv('iris.csv')

# 2. Lấy các cột đặc trưng và nhãn
X = df[['sepal_length', 'sepal_width', 'petal_length']].values  # (n, 3)
y = df['petal_width'].values  # (n, )

# 3. Chuẩn hoá dữ liệu (giúp thuật toán hội tụ nhanh hơn)
X = (X - X.mean(axis=0)) / X.std(axis=0)

# 4. Khởi tạo trọng số và siêu tham số
w = np.zeros(X.shape[1])  # 3 trọng số tương ứng 3 đặc trưng
b = 0.0
lr = 0.01
epochs = 200

# 5. Hàm tính MSE
def compute_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# 6. Huấn luyện bằng Gradient Descent
mse_history = []

for epoch in range(epochs):
    y_pred = np.dot(X, w) + b
    error = y - y_pred

    dw = -2 * np.dot(X.T, error) / len(X)
    db = -2 * np.mean(error)

    w -= lr * dw
    b -= lr * db

    mse = compute_mse(y, y_pred)
    mse_history.append(mse)

    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d}: MSE = {mse:.4f}, w = {w}, b = {b:.4f}")

# 7. Kết quả cuối cùng
print(f"\nMô hình huấn luyện xong:")
print(f"petal_width = {w[0]:.3f} * sepal_length + {w[1]:.3f} * sepal_width + {w[2]:.3f} * petal_length + {b:.3f}")

# 8. Vẽ đồ thị quá trình giảm MSE
plt.plot(mse_history)
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.title('Quá trình giảm MSE')
plt.grid()
plt.show()

from mpl_toolkits.mplot3d import Axes3D

# Chọn cố định sepal_width = 0 (giá trị chuẩn hóa)
fixed_sepal_width = 0

# Tạo lưới sepal_length và petal_length
x1_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 30)
x2_range = np.linspace(X[:, 2].min(), X[:, 2].max(), 30)
x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)

# Dự đoán petal_width từ mô hình đã học (với sepal_width cố định)
y_grid = (w[0] * x1_grid) + (w[1] * fixed_sepal_width) + (w[2] * x2_grid) + b

# Vẽ đồ thị 3D
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Vẽ điểm dữ liệu thật (với sepal_width gần 0 sau khi chuẩn hoá)
mask = np.abs(X[:,1] - fixed_sepal_width) < 0.1
ax.scatter(X[mask, 0], X[mask, 2], y[mask], color='blue', label='Dữ liệu thực')

# Vẽ mặt phẳng hồi quy
ax.plot_surface(x1_grid, x2_grid, y_grid, color='orange', alpha=0.6)

ax.set_xlabel('sepal_length (chuẩn hoá)')
ax.set_ylabel('petal_length (chuẩn hoá)')
ax.set_zlabel('petal_width')
ax.set_title('Mặt phẳng hồi quy tuyến tính (cố định sepal_width)')
plt.legend()
plt.show()
