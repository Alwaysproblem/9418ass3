import glob
import os
import numpy as np
import shutil

path = './conll_train'

dest = './conll_dev'

dir = os.path.dirname(__file__)

if path == './conll_train':
    x_file_list = glob.glob(os.path.join(dir, path + "/*.x"))
    y_file_list = glob.glob(os.path.join(dir, path + "/*.y"))

valid_ind = np.random.choice(range(len(x_file_list)), 2400, replace = False)

for ind in valid_ind:
    shutil.move(os.path.join(dir, x_file_list[ind]), os.path.join(dir, dest))
    shutil.move(os.path.join(dir, y_file_list[ind]), os.path.join(dir, dest))



