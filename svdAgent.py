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

def svd_on_columns(ratings_path):
    ratings_df = pd.read_csv(ratings_path, sep='\t', names=ratings_columns, engine='python')
    # Create a user-item ratings matrix
    ratings_matrix = ratings_df.pivot(index='user_id', 
        columns='movie_id', values='rating').fillna(0)
    ratings_sparse = csr_matrix(ratings_matrix.values)
    u, s, vt = svds(ratings_sparse, k=50)  # k is the number of factors
    sigma = np.diag(s)
    return u, sigma, vt


class AlgSVD(BaseAgent) : 
    def __init__(self, vehID: str, chan: Channel) -> None:
        self.true_u, self.true_sigma, self.true_vt = svd_on_columns(ratings_path) 
        
        self.a = None
        self.x = None
        self.Y = None 
        super().__init__(vehID, chan)
    
    def get_data(self):
        return (self.a, self.X, self.Y)
    
    def get_comm_data(self):
        return self.data_change

    def updateLocalData(self):
        return (1., 1., 1.)
    

    def aggregate(self):
        return
        

    def test(self):
        return 1.
    

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
    