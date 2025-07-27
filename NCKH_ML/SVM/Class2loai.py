# xây dựng mô hình phân loại nhị phân sử dụng SVM ko dùng sk learn để phân loại 2 loài hoa setosa và versicolor
# dựa trên 2 đặc trưng chiều dài và chiều rộng

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('iris.csv')

# 2. Lấy 2 lớp: setosa & versicolor
data = data[data["species"].isin(["setosa", "versicolor" ])]

# 3. Gán nhãn: setosa = -1, versicolor = +1
data["label"] = data["species"].apply(lambda x: -1 if x == "setosa" else 1)

# 4. Chọn 2 đặc trưng: petal length & petal width
X = data[["petal_length", "petal_width"]].values
y = data["label"].values

# 5. Chuẩn hoá (nếu muốn dễ tối ưu hơn)
X = (X - X.mean(axis=0)) / X.std(axis=0)

# 6. Khởi tạo tham số
w = np.zeros(X.shape[1])  # 2 chiều
b = 0
lr = 0.01
epochs = 1000
C = 1.0  # hệ số điều chỉnh soft margin (trade-off)

# 7. Huấn luyện SVM bằng SGD và hinge loss
for epoch in range(epochs):
    for i in range(len(X)):
        condition = y[i] * (np.dot(X[i], w) + b)
        if condition >= 1:
            # Không vi phạm margin → chỉ cập nhật regularization
            w -= lr * (2 * w)
        else:
            # Vi phạm margin → cập nhật cả loss + regularization
            w -= lr * (2 * w - C * y[i] * X[i])
            b -= lr * (-C * y[i])

# 8. Vẽ đường biên quyết định
x_line = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
y_line = -(w[0] * x_line + b) / w[1]

plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolors="k")
plt.plot(x_line, y_line, "k-")
plt.xlabel("Petal length (normalized)")
plt.ylabel("Petal width (normalized)")
plt.title("SVM Decision Boundary (Iris)")
plt.grid(True)
plt.show()
