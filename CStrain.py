import os
import glob
import numpy as np
# from queue import Queue

path = None
np.random.seed(1)

D = 2035523
# define the dimentional of give example sparse matrix.

Comp_dims = 200
#define the dimentional of compressed matrix.

A = np.mat(np.random.randn(Comp_dims, D))
# y = A * s, the s is sparse-matrix. the y is compressed measurements.

def Sparse_form(X, y, len_y, D):
    sparseMat = np.zeros((len_y, D))
    for i, j, v in X:
        sparseMat[i-1, j-1] = v
    return sparseMat

def Load_Data(path='./conll_train', sample = None):
    x_comp_data = []
    y_label = []
    print("Loading train Data...")
    dir = os.path.dirname(__file__)

    x_file_list = glob.glob(os.path.join(dir, path + "/*.x"))
    y_file_list = glob.glob(os.path.join(dir, path + "/*.y"))

    print(f"Parsing {len(x_file_list)} xfiles and {len(y_file_list)} yfiles.")

    if sample == None:
        for X, y in list(zip(x_file_list, y_file_list)):
            len_of_y = 0
            with open(y, "r") as yf:
                y_content = [int(row.strip()) for row in yf.readlines()]
                len_of_y = len(y_content)
                y_label.append(y_content)

            with open(X, "r") as xf:
                compX = []
                content = [row.strip() for row in xf.readlines()]
                for ind in range(1, len_of_y + 1):
                    X_data = np.mat(np.zeros((Comp_dims, 1)))
                    for c in content:
                        tran = [int(t) for t in c.split()]
                        if tran[0] == ind:
                            X_data += A[:, tran[1]]
                    if compX == []:
                        compX = X_data
                    else:
                        compX = np.concatenate([compX, X_data], axis = 1)
                        """
                        the label is like:
                          0    2
                        the X data is like:
                        [[18.  2.]
                         [18.  2.]
                         [18.  2.]
                         [18.  2.]
                         [18.  2.]
                         [18.  2.]
                         [18.  2.]
                         [18.  2.]
                         [18.  2.]
                         [18.  2.]]
                        """
                x_comp_data.append(compX)
    return x_comp_data, y_label


def main():
    Load_Data()

if __name__ == '__main__':
    main()