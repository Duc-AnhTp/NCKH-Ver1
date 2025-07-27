import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
from sklearn.model_selection import train_test_split

# 1. Đọc dữ liệu
df = pd.read_csv("iris.csv")
X = df[['petal_width']].values
y = df['petal_length'].values

# 2. Chia tập train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. SVR tuyến tính với kernel tuyến tính: K(x, x') = x * x'
C = 1.0
epsilon = 0.2
n = len(X_train)

# Tạo kernel tuyến tính
K = X_train @ X_train.T

# Ma trận trong bài toán QP
P = np.vstack([
    np.hstack([K, -K]),
    np.hstack([-K, K])
])
P = matrix(P)

q = np.hstack([epsilon - y_train, epsilon + y_train])
q = matrix(q, tc='d')

# G ràng buộc: 0 <= alpha_i, alpha_i* <= C
G_std = np.vstack([
    np.eye(2 * n),
    -np.eye(2 * n)
])
h_std = np.hstack([
    C * np.ones(2 * n),
    np.zeros(2 * n)
])

G = matrix(G_std)
h = matrix(h_std)

# A^T (alpha_i - alpha_i*) = 0
A = np.hstack([np.ones(n), -np.ones(n)])
A = matrix(A, (1, 2 * n), 'd')
b = matrix(0.0)

# Giải bài toán tối ưu
solvers.options['show_progress'] = False
solution = solvers.qp(P, q, G, h, A, b)
alphas = np.array(solution['x']).flatten()

alpha = alphas[:n]
alpha_star = alphas[n:]
alpha_diff = alpha - alpha_star

# 4. Hệ số w, b
w = np.sum(alpha_diff[:, None] * X_train, axis=0)
# Tìm b từ support vector
support_indices = np.where((alpha > 1e-4) | (alpha_star > 1e-4))[0]
b = np.mean(y_train[support_indices] - (X_train[support_indices] @ w.T))

# 5. Hàm dự đoán
def predict(X):
    return X @ w.T + b

y_pred = predict(X_test)

# 6. Vẽ kết quả
sorted_idx = np.argsort(X_test.flatten())
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test[sorted_idx], y_pred[sorted_idx], color='red', label='Predicted (SVR)')
plt.xlabel('Petal_width')
plt.ylabel('Petal_length')
plt.title('Manual SVR on Iris Dataset (Linear Kernel)')
plt.legend()
plt.grid(True)
plt.show()
