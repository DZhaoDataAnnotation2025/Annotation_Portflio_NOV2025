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
