import os
import glob
import numpy as np
# from queue import Queue
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import accuracy_score
import tensorflow as tf

tarin_path = './conll_train'
dev_path = './conll_dev'

np.random.seed(1)

def Load_Data(A, Comp_dims, path='./conll_train', sample = None, replace = False):
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


def softmax_classfier(input_data, label, Comp_dims, class_num, X_test, y_test):
    # print(input_data.shape)
    # print(label.shape)
    input_in = tf.placeholder(tf.float32, shape=(None, Comp_dims), name="input_data")
    lab = tf.placeholder(tf.int32, shape=(None,), name="label")

    pred = tf.layers.dense(input_in, class_num, activation=tf.nn.softmax)
    # print(pred.get_shape())
    y_label = tf.one_hot(lab, class_num)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                        logits = pred, 
                        labels = y_label
                    )
    loss = tf.reduce_mean(cross_entropy, name = "loss")
    opt = tf.train.AdamOptimizer(0.1).minimize(loss)
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, 1), tf.argmax(y_label, 1)), dtype = tf.float32), name = "accuracy")

    init = tf.global_variables_initializer()

    with tf.Session() as s:
        s.run(init)
        s.run(opt, feed_dict={input_in: input_data, lab: label})
        train_acc = s.run(acc, feed_dict={input_in: input_data, lab: label})
        test_acc = s.run(acc, feed_dict={input_in: X_test, lab: y_test})

    return train_acc, test_acc


def main():

    D = 2035523
    # define the dimentional of give example sparse matrix.

    Comp_dims = 50
    #define the dimentional of compressed matrix.

    A = np.mat(np.random.randn(Comp_dims, D))
    # y = A * s, the s is sparse-matrix. the y is compressed measurements.

    C = 23

    X_train, y_train = Load_Data(A, Comp_dims, sample=0.005)
    X_dev, y_dev = Load_Data(A, Comp_dims, path=dev_path, sample=0.005)

    print("start training...")

    train_acc, dev_acc = softmax_classfier(X_train, y_train, Comp_dims, C, X_dev, y_dev)

    # kernel = 1.0 * RBF(Comp_dims)
    # gpc = GaussianProcessClassifier(kernel=kernel)
    # gpc.fit(X_train, y_train)

    # y_train_pred = gpc.predict(X_train)
    # train_acc = accuracy_score(y_train, y_train_pred)

    print(f"the train accuracy is {train_acc}")

    # y_dev_pred = gpc.predict(X_dev)
    # dev_acc = accuracy_score(y_dev, y_dev_pred)

    print(f"the cross validation accuracy is {dev_acc}")


if __name__ == '__main__':
    main()