import os
import glob
import numpy as np

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
    print(data_x[index - 1])
    print(data_y[index - 1])
    sparseMat = Sparse_form(data_x[index - 1], data_y[index - 1], len(data_y[index - 1]), D)
    return sparseMat

def Merge_Data_Mat(data_x, data_y, rate, D):
    length = round(len(data_y) * rate)
    random_list = random_list = np.random.randint(1,len(data_y),length)
    print(random_list)
    mergeMat = Sparse_form(data_x[random_list[0] - 1], data_y[random_list[0] - 1], len(data_y[random_list[0] - 1]), D)
    print(0)
    for index in range(1,len(random_list)):
        print(index)
        newMat = Sparse_form(data_x[random_list[index] - 1], data_y[random_list[index] - 1], len(data_y[random_list[index] - 1]), D)
        mergeMat = np.vstack((mergeMat, newMat))
    return mergeMat


data_x, data_y = Load_Data()
# print(data_x[23 - 1])
# print(data_y[23 - 1])
D = 2035523
# random_list = np.random.randint(1,8936,10)
# print(random_list)
# matrix = Select_Data_Mat(data_x,data_y,23,D)
rate = 0.001
merge_matrix = Merge_Data_Mat(data_x, data_y, rate, D)
# print(matrix)
print(merge_matrix)
print('finish')
# for i in data:
#     print(i)
# a = '7 40 1\n'
# c = tuple(int(j) for j in a.split())
# print(c)