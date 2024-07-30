import numpy as np
import pandas as pd
from scipy.linalg import svd, diagsvd
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

import pickle

# Set global font size
plt.rcParams.update({'font.size': 14})


class AdamOptimizer:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update(self, params, grads):
        if self.m is None:
            self.m = np.zeros_like(grads)
            self.v = np.zeros_like(grads)

        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)

        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        # +!!!!!, not -!!!!!!
        params += self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return params


def dataset_movielens_10K():

    # Load Movielens dataset
    # url = "http://files.grouplens.org/datasets/movielens/ml-100k/u.data"
    url = "../datasets/MovieLens-100K/u.data"
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

    return ratings_matrix


def dataset_movielens_1M():
    # Load Movielens dataset
    # url = "http://files.grouplens.org/datasets/movielens/ml-100k/u.data"
    url = "../datasets/MovieLens-1M/"
    ratings = pd.read_csv(url+'ratings.dat',
                        delimiter='::',
                        engine='python',
                        names=['user_id', 'movie_id', 'rating', 'timestamp'])
    # Load movies
    movies = pd.read_csv(url+'movies.dat',
                        delimiter='::',
                        engine='python',
                        encoding='ISO-8859-1',
                        names=['movie_id', 'title', 'genres'])
    # Load users
    users = pd.read_csv(url+'users.dat',
                        delimiter='::',
                        engine='python',
                        names=['user_id', 'gender', 'age', 'occupation', 'zip'])
    # Create the user-item interaction matrix
    user_item_matrix = ratings.pivot(index='user_id', columns='movie_id', values='rating')
    # Fill missing values with 0 (or another placeholder if preferred)
    user_item_matrix = user_item_matrix.fillna(0)
    return user_item_matrix


# Target: find X of shape m*k and Y of shape n * k so that XY^T is close to A
# k <= min(m, n)
def baseline(A, learning_rate = 0.01, delta_k = 0, T=10):

    accuracy = [0] * T 
    m = A.shape[0]
    n = A.shape[1]
    assert(m <= n)
    k = m - delta_k 

    X = np.random.uniform(0., 1., (m, k))
    Y = np.random.uniform(0., 1., (n, k))

    for iter in range(T):
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
        accuracy[iter] = np.linalg.norm(diff) # l2-norm: ∥a−b∥_2 

        print("Round #", iter, ": ", accuracy[iter])

    return accuracy


def our_method(A, learning_rate = 0.01, delta_k = 0, T=10):

    accuracy = [0] * T 
    m = A.shape[0]
    n = A.shape[1]
    assert(m <= n)
    k = m - delta_k 

    optimizer1 = AdamOptimizer(lr=learning_rate)
    optimizer2 = AdamOptimizer(lr=learning_rate)

    X = np.random.uniform(0., 1., (m, k))
    Y = np.random.uniform(0., 1., (n, k))

    for iter in range(T):
        Adot = A.copy()
        Xdot = X.copy()
        Ydot = Y.copy()

        for i in range(k):
            err = Adot - np.outer(X[:, i], Y[:, i])
            if (iter <= 1):
                Xdot[:, i] = X[:, i] + learning_rate * 0.01 * np.dot(err, Y[:, i])
                Ydot[:, i] = Y[:, i] + learning_rate * 0.01 * np.dot(np.transpose(err), X[:, i])
                optimizer1.update(X[:, i], np.dot(err, Y[:, i]))
                optimizer2.update(Y[:, i], np.dot(np.transpose(err), X[:, i]))
            else:
                Xdot[:, i] = optimizer1.update(X[:, i], np.dot(err, Y[:, i]))
                Ydot[:, i] = optimizer2.update(Y[:, i], np.dot(np.transpose(err), X[:, i]))
            Adot = Adot - np.outer(X[:, i], Y[:, i])

        X = Xdot
        Y = Ydot
        diff = A - np.dot(X, np.transpose(Y))
        accuracy[iter] = np.linalg.norm(diff) # l2-norm: ∥a−b∥_2 

        print("Round #", iter, ": ", accuracy[iter])

    return accuracy



def exp01(cached=False):
    A = dataset_movielens_10K()
    T = 100
    acc = np.zeros([3, T])
    if (not cached):
        acc[2] = baseline(A, learning_rate=0.00001,  delta_k = 0, T=T)
        acc[0] = our_method(A, learning_rate=0.01,  delta_k = 0, T=T)
        acc[1] = our_method(A, learning_rate=0.005, delta_k = 0, T=T)
        with open('svd-baseline-exp01.pkl', 'wb') as file:
            pickle.dump(acc, file)
    else:
        with open('svd-baseline-exp01.pkl', 'rb') as file:
            acc = pickle.load(file)

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.plot(list(range(T)), acc[0], label="Our method (lr = 0.01)", linestyle='-')
    ax.plot(list(range(T)), acc[1], label="Our method (lr = 0.001)", linestyle='--')
    ax.plot(list(range(T)), acc[2], label="Baseline (lr = 0.0001)", linestyle='-.')
    ax.legend()
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Iterations")
    
    plt.show()


def exp02(cached=False):
    A = dataset_movielens_10K()
    T = 200
    acc = np.zeros([4, T])
    if (not cached):
        acc[0] = our_method(A, learning_rate=0.01,  delta_k = 0, T=T)
        acc[1] = our_method(A, learning_rate=0.01,  delta_k = 1, T=T)
        acc[2] = our_method(A, learning_rate=0.01,  delta_k = 2, T=T)
        acc[3] = our_method(A, learning_rate=0.01,  delta_k = 3, T=T)
        with open('svd-baseline-exp02.pkl', 'wb') as file:
            pickle.dump(acc, file)
    else:
        with open('svd-baseline-exp02.pkl', 'rb') as file:
            acc = pickle.load(file)

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.plot(list(range(T)), acc[0], label="Our method (lr = 0.01, k=0)", linestyle='-')
    ax.plot(list(range(T)), acc[1], label="Our method (lr = 0.01, k=1)", linestyle='--')
    ax.plot(list(range(T)), acc[2], label="Our method (lr = 0.01, k=2)", linestyle='-.')
    ax.plot(list(range(T)), acc[3], label="Our method (lr = 0.01, k=3)", linestyle='dotted')

    ax.legend()
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Iterations")
    
    plt.show()


#exp01()
A = dataset_movielens_10K()
T = 40
our_method(A, learning_rate=0.001,  delta_k = 0, T=T)