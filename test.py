import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import os
import numpy as np


def cos_similarity(a, b):
    # Normalize the rows of A and B
    norm_A = a / np.linalg.norm(a, axis=1)[:, np.newaxis]
    norm_B = b / np.linalg.norm(b, axis=1)[:, np.newaxis]
    # Compute cosine similarity as dot products of normalized rows
    return np.sum(norm_A * norm_B, axis=1)

datapath = 'data/MovieLens-100K'
ratings_path = os.path.join(datapath, 'u.data')
ratings_columns = ['user_id', 'movie_id', 'rating', 'timestamp']

def load_matrix(ratings_path):
    ratings_df = pd.read_csv(ratings_path, sep='\t', names=ratings_columns, engine='python')
    # Create a user-item ratings matrix
    ratings_matrix = ratings_df.pivot(index='user_id', 
        columns='movie_id', values='rating').fillna(0)
    return ratings_matrix

learning_rate = 0.001

a = load_matrix(ratings_path).values
a_shape = a.shape 
# 943 * 1682 
# rank(aa) = 943

n = 943
m = 1682
k = 943

x = np.random.rand(n, k)
y = np.random.rand(m, k)

diff = []

for i in range(500):
    aa = a
    for l in range(k):
        xl = x[:, l].copy().reshape(n, 1)
        yl = y[:, l].copy().reshape(m, 1)
        xl.shape
        yl.shape 
        err = aa - np.dot(xl, np.transpose(yl))
        xl = xl + learning_rate * np.dot(err, yl)
        yl = yl + learning_rate * np.dot(np.transpose(err), xl)
        aa  = aa - np.dot(xl, np.transpose(yl))

        x[:, l] = xl.reshape(n)
        y[:, l] = yl.reshape(m)

    #r = np.sum(a - np.dot(x, np.transpose(y)))
    r = cos_similarity(a, np.dot(x, np.transpose(y)))
    print("round ", i, ", cos diff is:", r)
    diff.append(r)
