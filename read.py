import os
import glob
import numpy as np
from sklearn.decomposition import PCA

def Sparse_form(X, y, len_y, D):
    sparseMat = np.zeros((len_y, D))
    for i, j, v in X:
        sparseMat[i-1, j-1] = v
    return sparseMat

def Load_Data(path='./conll_train'):
    print("Loading IMDB Data...")
    data_x = []
    data_y = []
    dir = os.path.dirname(__file__)
    file_list = glob.glob(os.path.join(dir, path + "/*"))
    file_list.sort(key = lambda x: int(x[16:-2]))
    print("Parsing %s files" % len(file_list))
    for i, f in enumerate(file_list):
        with open(f, "r",) as openf:
            s = openf.readlines()
            s_s = []
            if ' ' in s[0]:
                for line in s:
                    s_s.append(tuple(int(k) for k in line.split()))
                data_x.append(s_s)
            else:
                for line in s:
                    s_s.append(int(line))
                data_y.append(s_s)
    return data_x, data_y

def Select_Data_Mat(data_x, data_y, index, D):
    # print(data_x[index - 1])
    # print(data_y[index - 1])
    sparseMat = Sparse_form(data_x[index - 1], data_y[index - 1], len(data_y[index - 1]), D)
    return sparseMat

def Merge_Data_Mat(data_x, data_y, rate, D):
    length = round(len(data_y) * rate)
    random_list = random_list = np.random.randint(1,len(data_y),length)
    print(random_list)
    mergeMat = Sparse_form(data_x[random_list[0] - 1], data_y[random_list[0] - 1], len(data_y[random_list[0] - 1]), D)
    print(0)
    for index in range(1,len(random_list)):
        newMat = Sparse_form(data_x[random_list[index] - 1], data_y[random_list[index] - 1], len(data_y[random_list[index] - 1]), D)
        print('new')
        mergeMat = np.concatenate((mergeMat, newMat), axis = 0)
        print(index)
    return mergeMat

def PCA_data(x):
    pca = PCA(n_components=100)
    print(x.shape)
    pca.fit(x)
    data_x_pca = pca.transform(x)
    return data_x_pca

def PCA_All_Data_Mat(data_x,data_y,D):
    pca_mergeMat = np.array([0,0,0])
    while len(data_x) // 8 > 1:
        data_x_temp = data_x[:8]
        data_x = data_x[8:]
        data_y_temp = data_y[:8]
        data_y = data_y[8:]
        mergeMat = Sparse_form(data_x_temp[0], data_y_temp[0], len(data_y_temp[0]), D)
        for i in range(1,len(data_y_temp)):
            newMat = Sparse_form(data_x_temp[i], data_y_temp[i], len(data_y_temp[i]), D)
            print(i)
            mergeMat = np.concatenate((mergeMat, newMat), axis = 0)
            print('new_long')
        if pca_mergeMat.any() == 0:
            print('pca0_long')
            pca_mergeMat = PCA_data(mergeMat)
        else:
            print('pca1_long')
            pca_newMat = PCA_data(mergeMat)
            print('pca2_long')
            pca_mergeMat = np.concatenate((pca_mergeMat, pca_newMat), axis = 0)
    print(pca_mergeMat.shape)
    mergeMat = Sparse_form(data_x[0], data_y[0], len(data_y[0]), D)
    for i in range(1,len(data_y)):
        newMat = Sparse_form(data_x[i], data_y[i], len(data_y[i]), D)
        print(i)
        mergeMat = np.concatenate((mergeMat, newMat), axis = 0)
        print('new')
    if pca_mergeMat.any() == 0:
        print('pca0')
        pca_mergeMat = PCA_data(mergeMat)
    else:
        print('pca1')
        pca_newMat = PCA_data(mergeMat)
        print('pca2')
        pca_mergeMat = np.concatenate((pca_mergeMat, pca_newMat), axis = 0)
    return pca_mergeMat


data_x, data_y = Load_Data()

D = 2035523
# print(data_x[23 - 1])
# print(data_y[23 - 1])
data_x_pca = PCA_All_Data_Mat(data_x[:23],data_y[:23],D)
print(data_x_pca.shape)
# random_list = np.random.randint(1,8936,10)
# print(random_list)
# matrix = Select_Data_Mat(data_x,data_y,23,D)
# rate = 0.01
# merge_matrix = Merge_Data_Mat(data_x, data_y, rate, D)
# print(matrix)
# print(merge_matrix)
print(data_x_pca)
print('finish')
# for i in data:
#     print(i)
# a = '7 40 1\n'
# c = tuple(int(j) for j in a.split())
# print(c)