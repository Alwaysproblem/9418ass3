import gpflow
import tensorflow as tf
import numpy as np


def GPMC(X, y, X_test, C_num, start = 1):
    """
    the X should like: (batch_size, dims)
    the y should like: (batch_size, 1) and start with 0 not 1
    """
    dims = X.shape[1]
    y = y - start
    SVGP = gpflow.models.SVGP(
        X, y, kern=gpflow.kernels.RBF(dims) + gpflow.kernels.White(dims, variance=0.01), Z=X.copy(),
        likelihood=gpflow.likelihoods.MultiClass(3), num_latent=3, whiten=True, q_diag=True)

    opt = gpflow.train.ScipyOptimizer()
    opt.minimize(SVGP)
    p, _ = SVGP.predict_y(X_test)

    pred = np.argmax(p, axis=1)

    return pred + start


X = np.array([[1, 2, 3, 4], [4, 5, 6, 7], [8, 9, 10, 11]], dtype = np.float64)
y = np.array([[1], [2], [3]], dtype = np.float64)

X_test = np.array([[8, 9, 10, 11], [4, 5, 6, 7]], dtype = np.float64)

# SVGP = gpflow.models.SVGP(
#     X, y, kern=gpflow.kernels.RBF(4) + gpflow.kernels.White(4, variance=0.01), Z=X[:, :].copy(),
#     likelihood=gpflow.likelihoods.MultiClass(3), num_latent=3, whiten=True, q_diag=True)

# opt = gpflow.train.ScipyOptimizer()
# opt.minimize(SVGP)
# p, _ = SVGP.predict_y(X_test)

# print(SVGP.as_pandas_table())
# print(p)

pred = GPMC(X, y, X_test, 3)
print(pred)
