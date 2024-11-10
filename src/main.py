# Import necessary libraries
import numpy as np  # NumPy for efficient numerical computations
import time  # time for measuring execution time of operations

# Generate two large random arrays (vectors) with 1,000,000 elements each
# These represent sample data or weights in a deep learning model
arrayA = np.random.rand(1000000)
arrayB = np.random.rand(1000000)

# Record the start time before performing the dot product operation
tic = time.time()

# Compute the dot product of the two arrays
# np.dot is a highly optimized function for vector and matrix operations
# In deep learning, this operation is common for calculating neuron activations
c = np.dot(arrayA, arrayB)

# Record the end time after the computation is done
toc = time.time()

# Calculate and display the elapsed time in milliseconds
# Converting to milliseconds for easier interpretation of performance
print("Vectorized version: " + str(1000 * (toc - tic)) + " ms")

# Note:
# - This vectorized approach is highly efficient compared to a for-loop-based implementation.
# - Leveraging vectorized operations is critical for performance, especially with large datasets in deep learning.
# - The time difference shown here illustrates the speed advantage of vectorization, which is why it's a best practice.
