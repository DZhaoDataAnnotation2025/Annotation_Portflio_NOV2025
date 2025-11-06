# Math_Terms_Nov2025.py
import numpy as np
# Sigmoid
z = np.array([0, 1, -1])
y_hat = 1 / (1 + np.exp(-z))
print(f"Sigmoid: {y_hat}")
# Sigmoid with Positive z
z_pos = np.array([2, 3, 4])
y_hat_pos = 1 / (1 + np.exp(-z_pos))
print(f"Sigmoid with Positive z: {y_hat_pos}")
# Log Loss
import numpy as np
y, y_hat = 1, 0.7311
loss = -np.log(y_hat)
print(f"Log Loss: {loss:.4f}")
# Gradient Calculation
import numpy as np
y_hat, y, x1 = 0.7311, 1, 2
gradient = (y_hat - y) * x1
print(f"Gradient: {gradient:.4f}")
# Buffer Task Answers
print("1. Sigmoid for z=0: 0.5")
print("2. Log loss is high because 0.1 is a confident wrong prediction for y=1.")
print("3. Eta (η) is the step size in gradient descent.")
print("4. Shape of A·B: 2×2")
print("5. Log loss measures prediction error, penalizing confident mistakes.")
