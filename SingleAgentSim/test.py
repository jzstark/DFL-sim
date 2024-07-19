import numpy as np
import pandas as pd
from scipy.linalg import svd, diagsvd
from scipy.sparse import csr_matrix

# Load Movielens dataset
url = "http://files.grouplens.org/datasets/movielens/ml-100k/u.data"
columns = ['user_id', 'item_id', 'rating', 'timestamp']
data = pd.read_csv(url, sep='\t', names=columns)

# Create a user-item matrix
num_users = data['user_id'].nunique()
num_items = data['item_id'].nunique()

assert(num_users <= num_items)

#ratings_matrix = csr_matrix((data['rating'], (data['user_id'] - 1, data['item_id'] - 1)), shape=(num_users, num_items))
user_item_matrix = data.pivot(index='user_id', columns='item_id', values='rating')
user_item_matrix = user_item_matrix.fillna(0)
ratings_matrix = user_item_matrix.to_numpy()

# Simulate the partitioning of the matrix across 4 nodes
num_nodes = 4
cols_per_node = num_items // num_nodes
X = [ratings_matrix[:,i*cols_per_node:(i+1)*cols_per_node] for i in range(num_nodes)]
#HACK!!!!!!
num_items = cols_per_node * num_nodes
ratings_matrix_reduced = ratings_matrix[:, :num_items]

V = [np.zeros([num_items, X[i].shape[1]]) for i in range(num_nodes)]

m = num_users 

def householder_reflector(h, x):
    # Step 1: Define the vector x
    h = np.array(h, dtype=float)
    # Step 2: Compute alpha
    alpha = -np.sign(h[0]) * np.linalg.norm(h)
    # Step 3: Compute the vector u
    e1 = np.zeros_like(h)
    e1[0] = 1
    v = (h - alpha * e1) / np.linalg.norm(h - alpha * e1)
    #v = h - np.linalg.norm(h) * e1
    # Step 4: Construct the Householder matrix H
    H = np.eye(len(x)) - 2 * np.outer(v, v)
    return np.dot(H, x).copy()


def bSVD(md, sd):
    assert(len(md) == num_users)
    A  = np.diag(md) + np.diag(sd, 1)
    B = np.zeros((num_users, num_items - num_users))
    A = np.hstack((A, B))
    U, s, VT = svd(A)
    return U, diagsvd(s, num_users, num_items), VT

U = [ np.identity(m) ] * num_nodes

for i in range(m-2):
    h = 0
    for pid in range(num_nodes):
        h = h + np.dot(X[pid][i], np.transpose(X[pid][i+1:]))
    
    for pid in range(num_nodes):
        X[pid][i+1:] = householder_reflector(h, X[pid][i+1:])
        U[pid][i+1:] = householder_reflector(h, U[pid][i+1:])
    

alpha = np.zeros((num_nodes, m))
beta  = np.zeros((num_nodes, m - 1))


tmp = 0
for pid in range(num_nodes):
    tmp += np.linalg.norm(X[pid][0]) ** 2
tmp = np.sqrt(tmp)

for pid in range(num_nodes):
    alpha[pid][0] = tmp
    V[pid][0] = X[pid][0] / alpha[pid][0]


for i in range(1, m):
    tmp = 0
    for pid in range(num_nodes):
        tmp += np.dot(X[pid][i], np.transpose(V[pid][i-1]))
    for pid in range(num_nodes):
        beta[pid][i-1] = tmp
    
    for pid in range(num_nodes):
        X[pid][i] = X[pid][i] - beta[pid][i-1] * V[pid][i-1]
    
    tmp = 0
    for pid in range(num_nodes):
        tmp += np.linalg.norm(X[pid][i]) ** 2
    tmp = np.sqrt(tmp)

    for pid in range(num_nodes):
        alpha[pid][i] = tmp
        V[pid][i] = X[pid][i] / alpha[pid][i]

print(alpha, beta)

ub, sigma, vtb = bSVD(alpha[0], beta[0])


u  = np.dot(U[0], ub)

for pid in range(num_nodes):
    V[pid] = np.dot(vtb, V[pid]) # actually the vtb should be different on different nodes

"""
u = U[0] * ub
for pid in range(num_nodes):
    V[pid] = vtb[:, pid * cols_per_node : (pid+1)*cols_per_node] * V[pid]
"""


U_final = u
Sigma_final = sigma
VT_final = np.hstack(V)

diff = ratings_matrix_reduced - np.dot(np.dot(U_final, Sigma_final), VT_final)
error1 = np.linalg.norm(diff) # l2-norm: ∥a−b∥_2  
error2 = np.sum(np.abs(diff))