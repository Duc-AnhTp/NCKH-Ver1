import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 1. Load dữ liệu
df = pd.read_csv("iris.csv")

# 2. Gán nhãn số: setosa=0, versicolor=1, virginica=2
label_map = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
df['label'] = df['species'].map(label_map)

# 3. Chọn đặc trưng
X = df[['petal_length', 'petal_width']].values
y = df['label'].values

# 4. Chuẩn hoá
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Chia train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 6. Huấn luyện mô hình SVM (đa lớp tự động)
model = svm.SVC(kernel='linear', C=1.0, decision_function_shape='ovr')  # ovO cũng được
model.fit(X_train, y_train)

# 7. Dự đoán và đánh giá
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Độ chính xác: {acc:.2f}")

# 8. Vẽ đường biên phân lớp
# Tạo lưới điểm để tô màu vùng phân lớp
x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                     np.linspace(y_min, y_max, 500))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap='Pastel1')
scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='Set1', edgecolors='k')

# 9. Gắn nhãn
plt.xlabel("Petal Length (normalized)")
plt.ylabel("Petal Width (normalized)")
plt.title("SVM đa lớp với sklearn (Iris)")
plt.legend(handles=scatter.legend_elements()[0], labels=['Setosa', 'Versicolor', 'Virginica'])
plt.grid(True)
plt.show()
