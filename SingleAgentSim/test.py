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
X = [ratings_matrix[:,i*cols_per_node:(i+1)*cols_per_node] for i in range(num_nodes)]
#HACK!!!!!!
num_items = cols_per_node * num_nodes
ratings_matrix_reduced = ratings_matrix[:, :num_items]

V = [np.zeros([num_items, X[i].shape[1]]) for i in range(num_nodes)]

m = num_users 

def householder_reflector(h):
    # Step 1: Define the vector
    h = np.array(h, dtype=float)
    # Step 2: Compute alpha
    alpha = -np.sign(h[0]) * np.linalg.norm(h, ord=2)
    # Step 3: Compute the vector u
    e1 = np.zeros_like(h)
    e1[0] = 1
    v = (h - alpha * e1) / np.linalg.norm(h - alpha * e1, ord=2)
    #v = h - np.linalg.norm(h) * e1
    # Step 4: Construct the Householder matrix H
    return np.eye(len(h)) - 2 * np.outer(v, v)
    #return np.dot(H, x).copy()


## https://www.netlib.org/lapack/explore-html/d8/d0d/group__larfg_gadc154fac2a92ae4c7405169a9d1f5ae9.html#gadc154fac2a92ae4c7405169a9d1f5ae9
def householder_reflector3(h):
    h = np.array(h, dtype=float)
    v = h.copy()
    beta = -np.sign(h[0]) * np.linalg.norm(h, ord=2)
    tau = 1 - h[0] / beta
    v = v / (h[0] - beta)
    v[0] = 1.
    return np.eye(len(h)) - tau * np.outer(v, v)

## leads to nan errors
#def householder_reflector2(h, x): 
#    h = np.array(h, dtype=float)
#    e1 = np.zeros_like(h)
#    e1[0] = 1
#    house = h - np.linalg.norm(h, ord=2) * e1 
#    y = x.copy()
#    for j in range(x.shape[1]):
#        y[:, j] = y[:, j] * h
#    return y


def drotg(a, b):
    if b == 0:
        c = 1.0
        s = 0.0
        r = a
        z = 0.0
    elif a == 0:
        c = 0.0
        s = 1.0
        r = b
        z = 1.0
    else:
        r = np.hypot(a, b)
        c = a / r
        s = b / r
        if np.abs(a) > np.abs(b):
            z = s
        else:
            z = 1.0 / c

    return c, s, r, z



def bSVD(md, sd):
    #assert(len(md) == num_users)
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
        X[pid][i+1:] = np.dot(householder_reflector3(h), X[pid][i+1:])
        U[pid][i+1:] = np.dot(householder_reflector3(h), U[pid][i+1:])


"""

for(int i=0; i<bidiagonal_m-2; i++):


A vertical vector of length m - 1 - i is stored in the position ralha_bidiagonal_house+(i+1)*(m+1)


// Generate Householder

# compute I - tau.v.vt
# The reflector H is not stored as a full matrix. Instead, it is represented by the scalar τ and the vector v. 

LAPACKE_dlarfg(bidiagonal_m-1-i,  // length
    ralha_bidiagonal_house+(i+1)*(m+1), // first element; on exit, it stores value beta; does not matter here
    ralha_bidiagonal_house+(i+1)*(m+1) + m, // vector , results will be stored here
    m, //incx -- here it means vertically increase
    ralha_bidiagonal_tau+1+i); //scalar tau


ralha_bidiagonal_house[(i+1)*(m+1)] = 1.0;
// Apply Householder

# y <- alpha . A (size mxn). x  (size nx1) + beta . y 
# void cblas_dgemv(const CBLAS_LAYOUT layout, const CBLAS_TRANSPOSE trans, 
                 const int m, const int n, const double alpha, 
                 const double *A, const int lda, 
                 const double *x, const int incx, 
                 const double beta, double *y, const int incy);

                 
# vt_x <- A . x 
# vt_x = X[i+1:,:] . v_i (as calculated above: U'[i+1:, i+1] ralha_bidiagonal_house+(i+1)*(bidiagonal_m+1)
cblas_dgemv(CblasRowMajor, CblasTrans, 
    bidiagonal_m-1-i, bidiagonal_local_n, 1.0, 
    masked_x+(i+1)*bidiagonal_local_n, bidiagonal_local_n, 
    ralha_bidiagonal_house+(i+1)*(bidiagonal_m+1), bidiagonal_m, 
    0, vt_x, 1);

# X[i+1:] = X[i+1:] + alpha . ralha_bidiagonal_house+(i+1)*(bidiagonal_m+1)  . (vt_x ^T)
#void cblas_dger(const CBLAS_LAYOUT layout, const int m, const int n, 
#                const double alpha, 
#                const double *x, const int incx, 
#                const double *y, const int incy, 
#                double *A, 
#                const int lda);

X[i+1:] = X[i+1:] - tau_i * v_i * vt_x^T
cblas_dger(CblasRowMajor, bidiagonal_m-1-i, bidiagonal_local_n, 
    -1 * *(ralha_bidiagonal_tau+1+i),  // -1 * pointer  
    ralha_bidiagonal_house+(i+1)*(bidiagonal_m+1), bidiagonal_m, 
    vt_x, 1, 
    masked_x+(i+1)*bidiagonal_local_n, 
    bidiagonal_local_n);


# void cblas_drotg(double *a, double *b, double *c, double *s);
#a: Pointer to a double representing the first component of the vector. On exit, it is overwritten with the value r.
#b: Pointer to a double representing the second component of the vector. On exit, it is overwritten with the value z.
#c: Pointer to a double representing the cosine of the Givens rotation.
#s: Pointer to a double representing the sine of the Givens rotation.
    
    double beta_utils
    double *cs = new double[bidiagonal_m*2]();
    for(int i=0; i<bidiagonal_m-1; i++){
        beta_utils = x_upper_diagonal[i];
        cblas_drotg(x_diagonal+i, &beta_utils, cs+i*2, cs+i*2+1);
        x_upper_diagonal[i] = *(cs+i*2+1) * x_diagonal[i+1];
        x_diagonal[i+1] = *(cs+i*2) * x_diagonal[i+1];
    }
    

"""

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
        #NOTE: the source code looks like += but paper says "="!
        V[pid][i] += X[pid][i] / alpha[pid][i]

#TODO: According to the source code, tHere is a step of Given's Rotation here to update alpha and beta!!!

print(alpha, beta)

def process_a_b(alpha, beta):
    alpha = alpha.copy()
    beta = beta.copy()
    assert(len(alpha) - len(beta) == 1)
    for i in range(num_users - 1):
        c, s, r, _z = drotg(alpha[i], beta[i])
        alpha[i] = r
        beta[i]  = s * alpha[i+1]
        alpha[i+1] = c * alpha[i+1]
    return alpha, beta 

alpha1, beta1 = process_a_b(alpha[0], beta[0])
print(alpha1, beta1)

#ub, sigma, vtb = bSVD(alpha[0], beta[0])
ub, sigma, vtb = bSVD(alpha1, beta1)


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