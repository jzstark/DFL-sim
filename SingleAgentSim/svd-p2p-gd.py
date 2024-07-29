import numpy as np
import pandas as pd
from scipy.linalg import svd, diagsvd
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import random 

import pickle

# Set global font size
plt.rcParams.update({'font.size': 14})


def dataset_movielens_10K(n=1):

    # Load Movielens dataset
    # url = "http://files.grouplens.org/datasets/movielens/ml-100k/u.data"
    url = "../datasets/MovieLens-100K/u.data"
    columns = ['user_id', 'item_id', 'rating', 'timestamp']
    data = pd.read_csv(url, sep='\t', names=columns)

    # Create a user-item matrix
    num_users = data['user_id'].nunique()
    num_items = data['item_id'].nunique()

    #ratings_matrix = csr_matrix((data['rating'], (data['user_id'] - 1, data['item_id'] - 1)), shape=(num_users, num_items))
    user_item_matrix = data.pivot(index='user_id', columns='item_id', values='rating')
    user_item_matrix = user_item_matrix.fillna(0)
    ratings_matrix = user_item_matrix.to_numpy()

    rows_per_node = num_users // n
    X = [ratings_matrix[i*rows_per_node:(i+1)*rows_per_node, :] for i in range(n)]
    #HACK!!!!!!
    #num_items = rows_per_node * n
    #ratings_matrix_reduced = ratings_matrix[:num_items, :]

    return X, rows_per_node * n, num_items


def dataset_movielens_1M(n=1):
    # Load Movielens dataset
    # url = "http://files.grouplens.org/datasets/movielens/ml-100k/u.data"
    url = "../datasets/MovieLens-1M/"
    ratings = pd.read_csv(url+'ratings.dat',
                        delimiter='::',
                        engine='python',
                        names=['user_id', 'movie_id', 'rating', 'timestamp'])
    # Load movies
    #movies = pd.read_csv(url+'movies.dat',
    #                    delimiter='::',
    #                    engine='python',
    #                    encoding='ISO-8859-1',
    #                    names=['movie_id', 'title', 'genres'])
    ## Load users
    #users = pd.read_csv(url+'users.dat',
    #                    delimiter='::',
    #                    engine='python',
    #                    names=['user_id', 'gender', 'age', 'occupation', 'zip'])
    # Create the user-item interaction matrix
    user_item_matrix = ratings.pivot(index='user_id', columns='movie_id', values='rating')
    # Fill missing values with 0 (or another placeholder if preferred)
    user_item_matrix = user_item_matrix.fillna(0)
    ratings_matrix = user_item_matrix.to_numpy()

    num_users = ratings_matrix.shape[0]
    num_items = ratings_matrix.shape[1]
    rows_per_node = num_users // n
    X = [ratings_matrix[i*rows_per_node:(i+1)*rows_per_node, :] for i in range(n)]

    return X,  rows_per_node * n, num_items


def update(Y, xi, Ai, learning_rate=0.00001):
    Y = Y.copy()
    xi = xi.copy()
    Ai = Ai.copy()
    k = Y.shape[1]
    for l in range(k):
        xl = np.reshape(xi[:,l],[-1,1])
        yl = np.reshape(Y[:,l],[-1,1])
        err = Ai - np.matmul(xl, np.transpose(yl))
        xl_new = xl + (learning_rate * np.matmul(err, yl))
        yl_new = yl + (learning_rate * np.matmul(np.transpose(err), xl))
        xi[:,l] = np.reshape(xl_new, [-1])
        Y[:,l] = np.reshape(yl_new, [-1])
        Ai = Ai - np.matmul(xl_new,np.transpose(yl_new))
    return Y, xi



def eval(iters, num_agents=4, dataset_name='10K', k_fix = 0, neigh_num_fix = 0):
    if dataset_name == '10K':
        A, M, N  = dataset_movielens_10K(n=num_agents)
    else:
        A, M, N  = dataset_movielens_1M(n=num_agents)
    
    k = min(M, N) - k_fix
    X = []
    Y = []
    for ai in A:
        (mi, _) = ai.shape
        xi = np.random.uniform(0., 1., (mi, k))
        y  = np.random.uniform(0., 1., (N, k))
        X.append(xi)
        Y.append(y)
    
    accuracy = [0] * iters
    
    for iter in range(iters):
        neighbors = list(range(num_agents))
        for agent in range(num_agents):
            lst = neighbors.copy()
            lst.remove(agent)
            lst = random.sample(lst, num_agents - 1 - neigh_num_fix)
            for nei in lst:
                Y[nei], X[nei] = update(Y[agent], X[nei], A[nei])
        
        A_big = np.vstack(A)
        X_big = np.vstack(X)
        diff = A_big - np.dot(X_big, np.transpose(Y[0]))
        accuracy[iter] = np.linalg.norm(diff) # l2-norm: ∥a−b∥_2 
        print("Round #", iter, ": ", accuracy[iter])
    
    filename = "p2p_iter%d_agents%d_%s_kfix%d_neighborfix%d.pkl" % (iters, num_agents, dataset_name, k_fix, neigh_num_fix)
    return accuracy, filename
    

    
# acquire data

def eval_save(iters, num_agents=4, dataset_name='10K', k_fix = 0, neigh_num_fix = 0):
    try:
        acc, filename = eval(iters, num_agents, dataset_name, k_fix, neigh_num_fix)
        with open(filename, 'wb') as file:
            pickle.dump(acc, file)
    except Exception as e:
        print(f"Caught an exception in evaluation: {e}")

eval_save(100, num_agents=4,  dataset_name='10K', k_fix=0, neigh_num_fix=0)
eval_save(100, num_agents=6,  dataset_name='10K', k_fix=0, neigh_num_fix=0)
eval_save(100, num_agents=8,  dataset_name='10K', k_fix=0, neigh_num_fix=0)
#eval_save(100, num_agents=10, dataset_name='10K', k_fix=0, neigh_num_fix=0)

eval_save(100, num_agents=4,  dataset_name='10K', k_fix=10, neigh_num_fix=0)
eval_save(100, num_agents=6,  dataset_name='10K', k_fix=10, neigh_num_fix=0)
eval_save(100, num_agents=8,  dataset_name='10K', k_fix=10, neigh_num_fix=0)
#eval_save(100, num_agents=10, dataset_name='10K', k_fix=10, neigh_num_fix=0)

eval_save(100, num_agents=4,  dataset_name='10K', k_fix=20, neigh_num_fix=0)
eval_save(100, num_agents=6,  dataset_name='10K', k_fix=20, neigh_num_fix=0)
eval_save(100, num_agents=8,  dataset_name='10K', k_fix=20, neigh_num_fix=0)
#eval_save(100, num_agents=10, dataset_name='10K', k_fix=20, neigh_num_fix=0)

eval_save(100, num_agents=4,  dataset_name='10K', k_fix=0, neigh_num_fix=1)
eval_save(100, num_agents=6,  dataset_name='10K', k_fix=0, neigh_num_fix=1)
eval_save(100, num_agents=8,  dataset_name='10K', k_fix=0, neigh_num_fix=1)
#eval_save(100, num_agents=10, dataset_name='10K', k_fix=0, neigh_num_fix=1)
eval_save(100, num_agents=4,  dataset_name='10K', k_fix=10, neigh_num_fix=1)
eval_save(100, num_agents=6,  dataset_name='10K', k_fix=10, neigh_num_fix=1)
eval_save(100, num_agents=8,  dataset_name='10K', k_fix=10, neigh_num_fix=1)
#eval_save(100, num_agents=10, dataset_name='10K', k_fix=10, neigh_num_fix=1)
eval_save(100, num_agents=4,  dataset_name='10K', k_fix=20, neigh_num_fix=1)
eval_save(100, num_agents=6,  dataset_name='10K', k_fix=20, neigh_num_fix=1)
eval_save(100, num_agents=8,  dataset_name='10K', k_fix=20, neigh_num_fix=1)
#eval_save(100, num_agents=10, dataset_name='10K', k_fix=20, neigh_num_fix=1)


eval_save(100, num_agents=4,  dataset_name='10K', k_fix=0, neigh_num_fix=2)
eval_save(100, num_agents=6,  dataset_name='10K', k_fix=0, neigh_num_fix=2)
eval_save(100, num_agents=8,  dataset_name='10K', k_fix=0, neigh_num_fix=2)
#eval_save(100, num_agents=10, dataset_name='10K', k_fix=0, neigh_num_fix=2)
eval_save(100, num_agents=4,  dataset_name='10K', k_fix=10, neigh_num_fix=2)
eval_save(100, num_agents=6,  dataset_name='10K', k_fix=10, neigh_num_fix=2)
eval_save(100, num_agents=8,  dataset_name='10K', k_fix=10, neigh_num_fix=2)
#eval_save(100, num_agents=10, dataset_name='10K', k_fix=10, neigh_num_fix=2)
eval_save(100, num_agents=4,  dataset_name='10K', k_fix=20, neigh_num_fix=2)
eval_save(100, num_agents=6,  dataset_name='10K', k_fix=20, neigh_num_fix=2)
eval_save(100, num_agents=8,  dataset_name='10K', k_fix=20, neigh_num_fix=2)
#eval_save(100, num_agents=10, dataset_name='10K', k_fix=20, neigh_num_fix=2)

eval_save(100, num_agents=4,  dataset_name='1M', k_fix=0, neigh_num_fix=0)
eval_save(100, num_agents=6,  dataset_name='1M', k_fix=0, neigh_num_fix=0)
eval_save(100, num_agents=8,  dataset_name='1M', k_fix=0, neigh_num_fix=0)
#eval_save(100, num_agents=10, dataset_name='1M', k_fix=0, neigh_num_fix=0)

eval_save(100, num_agents=4,  dataset_name='1M', k_fix=10, neigh_num_fix=0)
eval_save(100, num_agents=6,  dataset_name='1M', k_fix=10, neigh_num_fix=0)
eval_save(100, num_agents=8,  dataset_name='1M', k_fix=10, neigh_num_fix=0)
#eval_save(100, num_agents=10, dataset_name='1M', k_fix=10, neigh_num_fix=0)

eval_save(100, num_agents=4,  dataset_name='1M', k_fix=20, neigh_num_fix=0)
eval_save(100, num_agents=6,  dataset_name='1M', k_fix=20, neigh_num_fix=0)
eval_save(100, num_agents=8,  dataset_name='1M', k_fix=20, neigh_num_fix=0)
#eval_save(100, num_agents=10, dataset_name='1M', k_fix=20, neigh_num_fix=0)

eval_save(100, num_agents=4,  dataset_name='1M', k_fix=0, neigh_num_fix=1)
eval_save(100, num_agents=6,  dataset_name='1M', k_fix=0, neigh_num_fix=1)
eval_save(100, num_agents=8,  dataset_name='1M', k_fix=0, neigh_num_fix=1)
#eval_save(100, num_agents=10, dataset_name='1M', k_fix=0, neigh_num_fix=1)
eval_save(100, num_agents=4,  dataset_name='1M', k_fix=10, neigh_num_fix=1)
eval_save(100, num_agents=6,  dataset_name='1M', k_fix=10, neigh_num_fix=1)
eval_save(100, num_agents=8,  dataset_name='1M', k_fix=10, neigh_num_fix=1)
#eval_save(100, num_agents=10, dataset_name='1M', k_fix=10, neigh_num_fix=1)
eval_save(100, num_agents=4,  dataset_name='1M', k_fix=20, neigh_num_fix=1)
eval_save(100, num_agents=6,  dataset_name='1M', k_fix=20, neigh_num_fix=1)
eval_save(100, num_agents=8,  dataset_name='1M', k_fix=20, neigh_num_fix=1)
#eval_save(100, num_agents=10, dataset_name='1M', k_fix=20, neigh_num_fix=1)


eval_save(100, num_agents=4,  dataset_name='1M', k_fix=0, neigh_num_fix=2)
eval_save(100, num_agents=6,  dataset_name='1M', k_fix=0, neigh_num_fix=2)
eval_save(100, num_agents=8,  dataset_name='1M', k_fix=0, neigh_num_fix=2)
#eval_save(100, num_agents=10, dataset_name='1M', k_fix=0, neigh_num_fix=2)
eval_save(100, num_agents=4,  dataset_name='1M', k_fix=10, neigh_num_fix=2)
eval_save(100, num_agents=6,  dataset_name='1M', k_fix=10, neigh_num_fix=2)
eval_save(100, num_agents=8,  dataset_name='1M', k_fix=10, neigh_num_fix=2)
#eval_save(100, num_agents=10, dataset_name='1M', k_fix=10, neigh_num_fix=2)
eval_save(100, num_agents=4,  dataset_name='1M', k_fix=20, neigh_num_fix=2)
eval_save(100, num_agents=6,  dataset_name='1M', k_fix=20, neigh_num_fix=2)
eval_save(100, num_agents=8,  dataset_name='1M', k_fix=20, neigh_num_fix=2)
#eval_save(100, num_agents=10, dataset_name='1M', k_fix=20, neigh_num_fix=2)

#eval_save(3, num_agents=10, dataset_name='1M', k_fix=20, neigh_num_fix=2)