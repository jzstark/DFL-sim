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
num_nodes = 1
cols_per_node = num_items // num_nodes
A = [ratings_matrix[:,i*cols_per_node:(i+1)*cols_per_node] for i in range(num_nodes)]
##HACK!!!!!! Each node contains the same number of columns
#num_items = cols_per_node * num_nodes
#X = ratings_matrix[:, :num_items]
A = A[0]

# Target: find X of shape m*k and Y of shape n * k so that XY^T is close to A
# k <= min(m, n)

learning_rate = 0.00001
m = num_users
n = num_items
k = m

X = np.random.uniform(0., 1., (m, k))
Y = np.random.uniform(0., 1., (n, k))


for iter in range(2000):

    Adot = A.copy()
    Xdot = X.copy()
    Ydot = Y.copy()

    for i in range(k):
        err = Adot - np.outer(X[:, i], Y[:, i])
        Xdot[:, i] = X[:, i] + learning_rate * np.dot(err, Y[:, i])
        Ydot[:, i] = Y[:, i] + learning_rate * np.dot(np.transpose(err), X[:, i])
        Adot = Adot - np.outer(X[:, i], Y[:, i])

    X = Xdot
    Y = Ydot

    diff = A - np.dot(X, np.transpose(Y))

    print("Round#", iter, ": ", np.linalg.norm(diff)) # l2-norm: ∥a−b∥_2  
    