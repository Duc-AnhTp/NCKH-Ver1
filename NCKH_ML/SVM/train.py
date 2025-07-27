import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 1. Đọc dữ liệu
data = pd.read_csv('iris.csv')

# 2. Lấy 2 lớp: setosa & versicolor
data = data[data["species"].isin(["setosa", "versicolor"])]

# 3. Gán nhãn: setosa = -1, versicolor = +1
data["label"] = data["species"].apply(lambda x: -1 if x == "setosa" else 1)

# 4. Chọn 2 đặc trưng
X = data[["petal_length", "petal_width"]].values
y = data["label"].values

# 5. Chia dữ liệu train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 6. Chuẩn hoá theo tập train
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std  # CHÚ Ý: chuẩn hoá test theo train

# 7. Khởi tạo tham số
w = np.zeros(X_train.shape[1])
b = 0
lr = 0.01
epochs = 1000
C = 1.0

# 8. Huấn luyện bằng SGD + hinge loss
for epoch in range(epochs):
    for i in range(len(X_train)):
        condition = y_train[i] * (np.dot(X_train[i], w) + b)
        if condition >= 1:
            w -= lr * (2 * w)
        else:
            w -= lr * (2 * w - C * y_train[i] * X_train[i])
            b -= lr * (-C * y_train[i])

# 9. Dự đoán trên tập test
y_pred = np.sign(np.dot(X_test, w) + b)

# 10. Tính độ chính xác
accuracy = np.mean(y_pred == y_test)
print("Độ chính xác trên tập test:", accuracy)

# 11. Vẽ biên quyết định (trên toàn bộ dữ liệu để minh họa)
X_all = np.vstack((X_train, X_test))
y_all = np.hstack((y_train, y_test))

x_line = np.linspace(X_all[:, 0].min(), X_all[:, 0].max(), 100)
y_line = -(w[0] * x_line + b) / w[1]

plt.scatter(X_all[:, 0], X_all[:, 1], c=y_all, cmap="bwr", edgecolors="k")
plt.plot(x_line, y_line, "k-", label="Decision boundary")
plt.xlabel("Petal length (normalized)")
plt.ylabel("Petal width (normalized)")
plt.title("SVM Decision Boundary (Train + Test)")
plt.grid(True)
plt.legend()
plt.show()
