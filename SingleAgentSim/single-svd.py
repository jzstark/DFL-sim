import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

# Load Movielens dataset
url = "http://files.grouplens.org/datasets/movielens/ml-100k/u.data"
columns = ['user_id', 'item_id', 'rating', 'timestamp']
data = pd.read_csv(url, sep='\t', names=columns)

# Create a user-item matrix
num_users = data['user_id'].nunique()
num_items = data['item_id'].nunique()
ratings_matrix = csr_matrix((data['rating'], (data['user_id'] - 1, data['item_id'] - 1)), shape=(num_users, num_items))

# Simulate the partitioning of the matrix across 4 nodes
num_nodes = 4
rows_per_node = num_users // num_nodes
partitions = [ratings_matrix[i*rows_per_node:(i+1)*rows_per_node] for i in range(num_nodes)]

def orthogonalize(matrix):
    """ Orthogonalize the columns of the input matrix using QR decomposition. """
    q, _ = np.linalg.qr(matrix)
    return q

def block_power_method(partitions, k=10, num_iter=10):
    num_nodes = len(partitions)
    local_Q = [np.random.rand(partition.shape[1], k) for partition in partitions]
    
    for _ in range(num_iter):
        # Local matrix multiplication
        local_Z = [partition.T @ (partition @ local_Q[i]) for i, partition in enumerate(partitions)]
        
        # Aggregate Z globally
        Z = sum(local_Z)
        
        # Compute SVD of Z
        U_Z, Sigma_Z, V_Z_T = np.linalg.svd(Z, full_matrices=False)
        
        # Update V globally
        V = V_Z_T.T
        
        print(np.shape(partitions[0]), np.shape(V))
        # Update Q locally
        local_Q = [(partition @ V) for partition in partitions]
        
        # Orthogonalize Q
        local_Q = [orthogonalize(q) for q in local_Q]
    
    return V, Sigma_Z, local_Q

# Run the block power method
k = 10  # Number of singular values/vectors to compute
V, Sigma, local_Q = block_power_method(partitions, k=k, num_iter=20)

# Compute final left singular vectors U
U_parts = [(partition @ V) @ np.diag(1 / Sigma) for partition in partitions]
U = np.vstack(U_parts)

# Construct the final U, Sigma, V matrices
U, Sigma, V = U, np.diag(Sigma), V

# Verify the reconstruction
original_matrix = ratings_matrix.toarray()
reconstructed_matrix_full = U @ Sigma @ V.T

# Calculate the Frobenius norm of the difference
error = np.linalg.norm(original_matrix - reconstructed_matrix_full, 'fro')

print("Frobenius norm of the difference:", error)

# Print results
print("U shape:", U.shape)
print("Sigma shape:", Sigma.shape)
print("V shape:", V.shape)