import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


df = pd.read_csv("iris.csv")

# Lọc 2 loài: setosa & versicolor
df = df[df['species'].isin(['setosa', 'versicolor'])]

# Gán nhãn: setosa = 0, versicolor = 1
df['label'] = df['species'].apply(lambda x: 0 if x == 'setosa' else 1)

# Chọn đặc trưng (feature)
X = df[['petal_length', 'petal_width']].values
y = df['label'].values

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Chia train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Huấn luyện mô hình SVM
model = svm.SVC(kernel='linear', C=1.0)  # Soft margin SVM
model.fit(X_train, y_train)

# Dự đoán
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Độ chính xác: {acc:.2f}")

# Vẽ đường biên phân lớp
w = model.coef_[0]
b = model.intercept_[0]

# Vẽ đường quyết định: w1*x + w2*y + b = 0 → y = -(w1*x + b)/w2
x_range = np.linspace(X_scaled[:, 0].min(), X_scaled[:, 0].max(), 100)
y_range = -(w[0]*x_range + b) / w[1]

plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='bwr', edgecolors='k')
plt.plot(x_range, y_range, 'k--')
plt.xlabel("Petal Length (normalized)")
plt.ylabel("Petal Width (normalized)")
plt.title("SVM with sklearn (Setosa vs Versicolor)")
plt.grid(True)
plt.show()
