import numpy as np


X = [(1, 1, 1), (1, 2, 1), (2, 3, 1), (2, 4, 1)]

y = [1, 2]

len_y = 2

D = 4


def Sparse_form(X, y, len_y, D):
    sparseMat = np.zeros((len_y, D))
    for i, j, v in X:
        sparseMat[i-1, j-1] = v
    return sparseMat


print(Sparse_form(X, y, len_y, D))
