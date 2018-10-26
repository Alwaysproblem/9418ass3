import numpy as np

test = ['1 1 1\n', '1 2 1\n', '1 58113 1\n', '1 703 1\n', '1 4 1\n', '1 58114 1\n', '1 5 1\n', '1 6 1\n', '1 265 1\n', '1 8 1\n', '1 9 1\n', '1 10 1\n', '1 1028 1\n', '1 5997 1\n', '1 13 1\n', '1 1030 1\n', '1 20778 1\n', '1 11731 1\n', '2 17 1\n', '2 58115 1\n']

Comp_dims = 10
len_of_y = 2

A = np.mat(np.ones((Comp_dims, 60000)))

content = [row.strip() for row in test]
compX = []

for ind in range(1, len_of_y + 1):
    X_data = np.mat(np.zeros((Comp_dims, 1)))
    for c in content:
        tran = [int(t) for t in c.split()]
        if tran[0] == ind:
            X_data += A[:, tran[1]]
    print(X_data)
    if compX == []:
        compX = X_data
    else:
        compX = np.concatenate([compX, X_data], axis = 1)

print(compX)