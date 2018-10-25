import os
import glob
import numpy as np
# from queue import Queue
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

tarin_path = './conll_train'
dev_path = './conll_dev'

np.random.seed(1)

D = 2035523
# define the dimentional of give example sparse matrix.

Comp_dims = 100
#define the dimentional of compressed matrix.

A = np.mat(np.random.randn(Comp_dims, D))
# y = A * s, the s is sparse-matrix. the y is compressed measurements.

def Load_Data(path='./conll_train', sample = None, replace = False):
    x_comp_data = []
    y_label = []
    name = path.split("_")[-1]
    print(f"Loading {name} Data...")
    dir = os.path.dirname(__file__)

    x_file_list = glob.glob(os.path.join(dir, path + "/*.x"))
    y_file_list = glob.glob(os.path.join(dir, path + "/*.y"))

    if sample == None:
        pass
    else:
        Num = len(x_file_list)
        sam_ind = np.random.choice(range(Num), round(sample * Num), replace = replace)
        x_file_list = [x_file_list[I] for I in sam_ind]
        y_file_list = [y_file_list[I] for I in sam_ind]
    
    print(f"Parsing {len(x_file_list)} xfiles and {len(y_file_list)} yfiles.")


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
                    the X data is like: shape like (Comp_dims, len of label)
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
            x_comp_data.append(compX.T)

    return np.concatenate(x_comp_data, axis = 0), np.concatenate(y_label, axis = 0)


def main():
    X_train, y_train = Load_Data(sample=0.1)
    kernel = 1.0 * RBF(Comp_dims)
    gpc = GaussianProcessClassifier(kernel=kernel)
    gpc.fit(X_train, y_train)
    train_acc = gpc.score(X_train, y_train)
    print(f"the train accuracy is {train_acc}")
    X_train, y_train = Load_Data()
    X_dev, y_dev = Load_Data(path=dev_path, sample=0.1)
    dev_acc = gpc.score(X_dev, y_dev)
    print(f"the train accuracy is {dev_acc}")


if __name__ == '__main__':
    main()