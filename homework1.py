# Srikanth Schelbert Homework1

import matplotlib.pyplot as plt
import numpy as np
import math

import functools

centerpoint = (0.3, 0.4)


def f(x, point):
    """Function for ring"""
    return math.exp(-100 * ((math.dist(x, point) - 0.2)**2))


# Generate grid points
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y)


# @functools.lru_cache(maxsize=None)
def compute_z(X, Y, centerpoint):
    """Compute function values Z"""
    Z = np.zeros_like(X)  # Initialize array for function values
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = f([X[i, j], Y[i, j]], centerpoint)
    return Z


### PART 1 ###
# Sample 100 random locations
np.random.seed(42)  # for reproducibility
num_samples = 100
sampled_locations = np.random.rand(num_samples, 2)

# Create the plot
plt.figure(figsize=(8, 6))
# Plot sampled locations
plt.scatter(centerpoint[0], centerpoint[1], marker='x',
            color='blue', label='centerpoint')

# results of sensor (0 neg, 1 pos)
results = []
for loc in sampled_locations:
    x, y = loc
    if np.random.rand() < f([x, y], centerpoint):
        plt.scatter(x, y, color='green')
        results.append(1)
    else:
        plt.scatter(x, y, color='red')
        results.append(0)

print(f"Sampled locs: {sampled_locations[:5]}")
print(f"Sensor reads: {results[:50]}")
# Create dummy scatter plots for legend
plt.scatter([], [], color='green', label='positive')
plt.scatter([], [], color='red', label='negative')

Z_mat = compute_z(X, Y, centerpoint)

### PART 2 ###
likelihood = 0


def prob_z_given_x(x_i, s_i, z_i):
    if z_i == 1:
        return f(x_i, s_i)
    else:
        return 1 - f(x_i, s_i)


# Add legend
plt.legend(loc='upper right')
plt.imshow(Z_mat, extent=[0, 1, 0, 1], cmap='gray',
           origin='lower', aspect='auto')
plt.colorbar(label='Function Value')
plt.title('Visualization of the Function')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# rand_points = []
# for i in range(100):
#     rand_point = [np.random.uniform(), np.random.uniform()]
#     rand_points.append(rand_point)
#     val = np.random.uniform()
#     if f(rand_point, centerpoint) < val:
