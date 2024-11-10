# Import necessary libraries
import numpy as np  # NumPy for efficient numerical computations
import time  # time for measuring execution time of operations

# --------------------------------------
# Vectorized Example
# --------------------------------------

# Generate two large random arrays (vectors) with 100,000 elements each
# In deep learning, these arrays could represent a large dataset or model weights
# Arrays of this size are typical in machine learning, where high-dimensional data is common
arrayA = np.random.rand(100000)
arrayB = np.random.rand(100000)

# Record the start time before performing the dot product operation
tic = time.time()

# Compute the dot product of the two arrays
# Using np.dot, a vectorized function that is highly optimized for speed and memory efficiency
# Vectorized operations are fast because they leverage low-level, highly optimized C libraries in NumPy,
# eliminating the need for explicit loops in Python and instead relying on compiled code.
# In deep learning, vectorization is commonly used for matrix multiplications,
# which are the core of operations in neural networks, especially for forward and backward propagation
c = np.dot(arrayA, arrayB)

# Record the end time after the computation is done
toc = time.time()

# Print the computed dot product result
print(c)

# Calculate and display the elapsed time in milliseconds
# The time difference shows how much faster the vectorized approach is, ideal for deep learning tasks
print("Vectorized version: " + str(1000 * (toc - tic)) + " ms")

# --------------------------------------
# Explanation of Vectorization
# --------------------------------------
# Why Vectorization is Important:
# In deep learning and machine learning, we work with large datasets and complex models
# that require significant computational resources.
# Vectorization allows us to perform batch operations on entire arrays or matrices,
# making the code both faster and more efficient. By leveraging vectorized operations,
# we minimize Python overhead and allow operations to run on optimized low-level implementations.
# This is essential in deep learning, where models may have millions of parameters.
# Operations like matrix multiplications (e.g., dot products) are core to neural networks,
# and vectorization makes them feasible in practice, as well as scalable for large data.

# --------------------------------------
# Non-Vectorized Example (for comparison)
# --------------------------------------

# Initialize the variable to store the result of the dot product
c = 0

# Record the start time before performing the loop-based dot product
tic = time.time()

# Perform the dot product manually using a for-loop (non-vectorized approach)
# Here, we are iterating over each element, which is much slower in Python
# because of the interpreter overhead. In deep learning, this approach is impractical
# for large-scale operations due to time constraints and inefficiency
for i in range(100000):
    c += arrayA[i] * arrayB[i]

# Record the end time after the computation is done
toc = time.time()

# Print the computed dot product result
print(c)

# Calculate and display the elapsed time in milliseconds
# The time difference between this and the vectorized approach highlights the efficiency of vectorization
print("Non-Vectorized version: " + str(1000 * (toc - tic)) + " ms")

# --------------------------------------
# Summary:
# - Vectorization is essential in deep learning because it significantly reduces computation time.
# - Non-vectorized approaches (for-loops) are slower and less efficient, especially for large datasets.
# - NumPy's vectorized functions leverage optimized C libraries and hardware acceleration,
#   which is especially beneficial when training deep learning models.
# - This code demonstrates that vectorized computations are much faster, highlighting why they
#   are preferred in deep learning frameworks like TensorFlow and PyTorch.
