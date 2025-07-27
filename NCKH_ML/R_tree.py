import pandas as pd

# Đọc dữ liệu Iris và chọn petal_length làm biến mục tiêu
iris = pd.read_csv('iris.csv')
X = iris[['sepal_length','sepal_width','petal_width']].values.tolist()
y = iris['petal_length'].tolist()

class DecisionTreeRegressorFromScratch:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
    
    def _variance(self, values):
        # Tính tổng bình phương sai lệch (impurity) cho tập giá trị
        if not values: return 0
        mean_val = sum(values) / len(values)
        return sum((v - mean_val)**2 for v in values)
    
    def _best_split(self, X, y):
        best_gain = 0
        best_feat, best_thresh = None, None
        current_var = self._variance(y)
        n_features = len(X[0])
        # Duyệt từng tính năng tương tự phần phân loại
        for feature in range(n_features):
            values = sorted(set(x[feature] for x in X))
            for i in range(1, len(values)):
                thresh = (values[i-1] + values[i]) / 2
                left_y = [y[j] for j,x in enumerate(X) if x[feature] <= thresh]
                right_y = [y[j] for j,x in enumerate(X) if x[feature] >  thresh]
                if not left_y or not right_y: 
                    continue
                # Tính phương sai con trái và phải
                var_left = self._variance(left_y)
                var_right = self._variance(right_y)
                gain = current_var - (var_left + var_right)
                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, feature, thresh
        return best_gain, best_feat, best_thresh
    
    def _build_tree(self, X, y, depth=0):
        # Nếu không chia tiếp được hoặc đạt chiều sâu tối đa, tạo lá với giá trị trung bình
        if depth == self.max_depth or len(y) < self.min_samples_split:
            return {'type': 'leaf', 'value': sum(y)/len(y)}
        gain, feature, thresh = self._best_split(X, y)
        if gain == 0 or feature is None:
            return {'type': 'leaf', 'value': sum(y)/len(y)}
        # Tạo cây con trái phải như phần phân loại
        left_X, left_y, right_X, right_y = [], [], [], []
        for xi, yi in zip(X, y):
            if xi[feature] <= thresh:
                left_X.append(xi); left_y.append(yi)
            else:
                right_X.append(xi); right_y.append(yi)
        left_tree  = self._build_tree(left_X, left_y, depth+1)
        right_tree = self._build_tree(right_X, right_y, depth+1)
        return {'type': 'node', 'feature': feature, 'threshold': thresh,
                'left': left_tree, 'right': right_tree}

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _predict_one(self, xi, node):
        if node['type'] == 'leaf':
            return node['value']
        if xi[node['feature']] <= node['threshold']:
            return self._predict_one(xi, node['left'])
        else:
            return self._predict_one(xi, node['right'])
    
    def predict(self, X):
        return [self._predict_one(xi, self.tree) for xi in X]

# Tạo và đánh giá mô hình cây hồi quy
reg = DecisionTreeRegressorFromScratch(max_depth=3)
reg.fit(X, y)
predictions_reg = reg.predict(X)
# Tính hệ số xác định R^2
ss_res = sum((yt - yp)**2 for yt, yp in zip(y, predictions_reg))
ss_tot = sum((yt - sum(y)/len(y))**2 for yt in y)
r2 = 1 - ss_res/ss_tot
print("R^2 (scratch):", r2)



import pandas as pd
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Đọc dữ liệu, dùng petal_length làm mục tiêu
iris = pd.read_csv('iris.csv')
X = iris[['sepal_length', 'sepal_width', 'petal_width']].values
y = iris['petal_length'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Xây dựng mô hình DecisionTreeRegressor
reg_lib = DecisionTreeRegressor(max_depth=3)
reg_lib.fit(X_train, y_train)

# Dự đoán và tính R^2
y_pred_reg = reg_lib.predict(X_test)
print("R^2 (sklearn - hồi quy):", r2_score(y_test, y_pred_reg))

# Vẽ cây hồi quy
plt.figure(figsize=(12, 6))
plot_tree(reg_lib,
          feature_names=['sepal_length', 'sepal_width', 'petal_width'],
          filled=True,
          rounded=True)
plt.title("Cây hồi quy Decision Tree")
plt.show()
