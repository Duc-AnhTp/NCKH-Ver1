import numpy as np
import matplotlib.pyplot as plt

# 1. Định nghĩa hàm loss
# Giả sử mình có hàm loss c(θ) = θ² - 8θ + 10
# Khai báo hàm loss
def cal_loss_function(theta):
    return theta**2 - 8*theta + 10

# Đạo hàm của hàm Loss L'(theta) = 2*theta - 8 (đạo hàm của theta^2 - 8theta + 10)
def cal_loss_derivative(theta):
    return 2*theta - 8

# 2. Vẽ đồ thị hàm loss
# Khai báo khoảng giá trị của theta
theta_values = np.linspace(-4, 12, 400)

# giá trị của hàm loss theo khoảng theta
loss_values = cal_loss_function(theta_values)

# Vẽ đồ thị hàm loss
plt.plot(theta_values, loss_values)

# Khai báo giá trị theta và siêu tham số
theta = 11
learning_rate = 0.1

# # cách 1 cài ngưỡng epsilon để dừng
# epsilon = 1e-4

# while True:
#     # tính giá trị đạo hàm
#     grad = cal_loss_derivative(theta)

#     if np.abs(grad) < epsilon:
#         break

#     loss = cal_loss_function(theta)

#     plt.plot(theta, loss, 'ro')

#     plt.pause(0.1)
#     theta = theta - learning_rate * grad

# print("Giá trị theta tối ưu:::", theta)
# plt.show()
# Cách 2: Dừng quá trình cập nhật theta sau 1 số vòng lặp nhất định
N = 100

for i in range(N):
    # tính giá trị đạo hàm
    grad = cal_loss_derivative(theta)

    # if np.abs(grad) < epsilon:
    # #   break

    loss = cal_loss_function(theta)

    plt.plot(theta, loss, 'ro')

    plt.pause(0.1)
    theta = theta - learning_rate*grad

print("Giá trị theta tối ưu:::", theta)
plt.show()