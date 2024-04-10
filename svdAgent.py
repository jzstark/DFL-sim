from base import BaseAgent, Message, Channel
import random
import math

import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import os
import numpy as np

datapath = 'data/MovieLens-100K'
ratings_path = os.path.join(datapath, 'u.data')
ratings_columns = ['user_id', 'movie_id', 'rating', 'timestamp']

def cos_similarity(a, b):
    # Normalize the rows of A and B
    norm_A = a / np.linalg.norm(a, axis=1)[:, np.newaxis]
    norm_B = b / np.linalg.norm(b, axis=1)[:, np.newaxis]
    return np.sum(norm_A * norm_B, axis=1)


def load_movielens_matrix(ratings_path=ratings_path):
    ratings_df = pd.read_csv(ratings_path, sep='\t', names=ratings_columns, engine='python')
    # Create a user-item ratings matrix
    ratings_matrix = ratings_df.pivot(index='user_id', 
        columns='movie_id', values='rating').fillna(0)
    return ratings_matrix.values


def svd_on_columns(ratings_path):
    ratings_matrix = load_movielens_matrix(ratings_path)
    ratings_sparse = csr_matrix(ratings_matrix)
    u, s, vt = svds(ratings_sparse, k=50)  # k is the number of factors
    sigma = np.diag(s)
    return u, sigma, vt


movie_matrix = load_movielens_matrix(ratings_path)


class SVDAgent(BaseAgent): 
    _id_counter = 0

    def __init__(self, vehID: str, chan: Channel) -> None:
        # n: total number of vehicles 
        # self.true_u, self.true_sigma, self.true_vt = svd_on_columns(ratings_path) 
        super().__init__(vehID, chan)
        #HACK:
        self.number_nodes = 96
        self.matrix_row, self.matrix_col = movie_matrix.shape[0], movie_matrix.shape[1]
        self.shard_size = self.matrix_row // self.number_nodes

        self.matrix = self._shard_data(SVDAgent._id_counter)
        self.r = self.matrix.shape[0]
        self.c = self.matrix.shape[1]
        # HACK!
        self.k = self.matrix_row
        self.X = np.random.uniform(low=0, high=1, size=[self.r, self.k])
        self.Y = np.random.uniform(low=0, high=1, size=[self.c, self.k])
        self.learning_rate = 0.001

        SVDAgent._id_counter += 1
    
    def _shard_data(self, i):
        s = i * self.shard_size
        #e = ((i+1) * self.shard_size, self.matrix_col)
        e = self.matrix_row if i == self.number_nodes - 1 else (i+1) * self.shard_size
        return movie_matrix[s:e].copy()
    
    def get_data(self):
        # "weights"
        return (self.X, self.Y)
    
    def get_comm_data(self):
        # send delta(Y)'s
        return self.data_change

    # Update weight; return deltaY
    def updateLocalData(self):
        Ai = self.matrix
        oldY = self.Y.copy()
        for l in range(self.k):
            xl = np.reshape(self.X[:,l],[-1,1])
            yl = np.reshape(self.Y[:,l],[-1,1])
            err = Ai - np.matmul(xl, np.transpose(yl))
            xl_new = xl + (self.learning_rate * np.matmul(err, yl))
            yl_new = yl + (self.learning_rate * np.matmul(np.transpose(err), xl))
            self.X[:,l] = np.reshape(xl_new, [-1])
            self.Y[:,l] = np.reshape(yl_new, [-1])
            Ai = Ai - np.matmul(xl_new,np.transpose(yl_new))
        return self.Y - oldY
        
    # change local data based on lists of delta(Y)
    def aggregate(self):
        datalist = self.flat_cached_data() # state_dicts 
        if datalist == []: return
        delta_y = np.sum(datalist, axis=0)
        self.Y += delta_y
        return
        

    def test(self):
        #TODO: Wrong 
        return cos_similarity(movie_matrix, np.dot(self.X, np.transpose(self.Y)))
    

    


class SimpleAgent(BaseAgent):
    def __init__(self, vehID: str, chan: Channel) -> None:
        super().__init__(vehID, chan) 
        self.v : float = 0.
        self.old_v: float = 0
    
    def get_data(self) -> Message:
        return self.v
    
    def updateLocalData(self):
        self.old_v = self.v 
        self.v += random.random()
    
    def aggregate(self):
        s = 0
        c = 1
        for d in self.flat_cached_data():
            s += d
            c += 1
        self.v = s / c

    def test(self):
        return self.v
    