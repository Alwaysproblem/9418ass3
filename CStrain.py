import os
import glob
import numpy as np
from sklearn.decomposition import PCA

path = None
np.random.seed(1)

D = 2035523
# define the dimentional of give example sparse matrix.

Comp_dims = 200
#define the dimentional of compressed matrix.

A = np.random.randn(Comp_dims, D)
# y = A * s, the s is sparse-matrix. the y is compressed measurements.


def Sparse_form(X, y, len_y, D):
    sparseMat = np.zeros((len_y, D))
    for i, j, v in X:
        sparseMat[i-1, j-1] = v
    return sparseMat

def Load_Data(path='./conll_train'):
    print("Loading train Data...")
    dir = os.path.dirname(__file__)
    x_file_list = glob.glob(os.path.join(dir, path + "/*.x"))
    y_file_list = glob.glob(os.path.join(dir, path + "/*.y"))

    print(f"Parsing {len(x_file_list)} xfiles and {len(y_file_list)} yfiles.")

    for X, y in list(zip(x_file_list, y_file_list)):
        with open(f, "r",) as xf:
            


    # print(x_file_list[3])
    # print(y_file_list[3])

    # x_file_list.sort(key = lambda x: int(x[16:-2]))
    # y_file_list.sort(key = lambda x: int(x[16:-2]))


    # print("Parsing %s files" % len(file_list))
    # for i, f in enumerate(file_list):
    #     with open(f, "r",) as openf:
    #         s = openf.readlines()
    #         s_s = []
    #         if ' ' in s[0]:
    #             for line in s:
    #                 s_s.append(tuple(int(k) for k in line.split()))
    #             data_x.append(s_s)
    #         else:
    #             for line in s:
    #                 s_s.append(int(line))
    #             data_y.append(s_s)
    # return data_x, data_y



def main():
    Load_Data()

if __name__ == '__main__':
    main()