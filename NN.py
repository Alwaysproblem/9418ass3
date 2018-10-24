import numpy as np
import tensorflow as tf
from read import *

D = 2035523
INPUT_in = tf.placeholder(name='INPUT_in', shape=(None,D),dtype=tf.float32)
OUTPUT_label = tf.placeholder(name='OUTPUT_label', shape=(None,D),dtype=tf.float32)
hidden_unit_list = [D,1000,300]

learning_rate = 0.001

def NNgraph(INPUT, hidden_unit_list, unit_size):
    with tf.variable_scope("Neural_Net1"):
        with tf.variable_scope("layer1"):
            W1 = tf.get_variable(
                    "W1", 
                    shape=(INPUT.get_shape()[1], hidden_unit_list[0]), 
                    initializer=tf.random_normal_initializer(0, 0.3)
                )
            b1 = tf.get_variable(
                    "b1", 
                    shape=(1, hidden_unit_list[0]), 
                    initializer=tf.constant_initializer(0.1)
                )
            layer1 = tf.nn.relu(tf.matmul(INPUT, W1) + b1)

        with tf.variable_scope("layer2"):
            W2 = tf.get_variable(
                    "W2", 
                    shape=(hidden_unit_list[0], hidden_unit_list[1]), 
                    initializer=tf.random_normal_initializer(0, 0.3)
                )
            b2 = tf.get_variable(
                    "b2", 
                    shape=(1, hidden_unit_list[1]), 
                    initializer=tf.constant_initializer(0.1)
                )
            layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2)

        with tf.variable_scope("layer3"):
            W3 = tf.get_variable(
                    "W3", 
                    shape=(hidden_unit_list[1], hidden_unit_list[2]), 
                    initializer=tf.random_normal_initializer(0, 0.3)
                )
            b3 = tf.get_variable(
                    "b3", 
                    shape=(1, hidden_unit_list[2]), 
                    initializer=tf.constant_initializer(0.1)
                )
            output = tf.matmul(layer2, W3) + b3

    return output

hidden_unit_list2 = [300,1000,D]
INPUT_in2 = tf.placeholder(name='INPUT_in', shape=(None,300),dtype=tf.float32)

def NNgraph2(INPUT2, hidden_unit_list2, unit_size):
    with tf.variable_scope("Neural_Net2"):
        with tf.variable_scope("layer1"):
            W1 = tf.get_variable(
                    "W1", 
                    shape=(INPUT2.get_shape()[1], hidden_unit_list2[0]), 
                    initializer=tf.random_normal_initializer(0, 0.3)
                )
            b1 = tf.get_variable(
                    "b1", 
                    shape=(1, hidden_unit_list2[0]), 
                    initializer=tf.constant_initializer(0.1)
                )
            layer1 = tf.nn.relu(tf.matmul(INPUT2, W1) + b1)

        with tf.variable_scope("layer2"):
            W2 = tf.get_variable(
                    "W2", 
                    shape=(hidden_unit_list2[0], hidden_unit_list2[1]), 
                    initializer=tf.random_normal_initializer(0, 0.3)
                )
            b2 = tf.get_variable(
                    "b2", 
                    shape=(1, hidden_unit_list2[1]), 
                    initializer=tf.constant_initializer(0.1)
                )
            layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2)

        with tf.variable_scope("layer3"):
            W3 = tf.get_variable(
                    "W3", 
                    shape=(hidden_unit_list2[1], hidden_unit_list2[2]), 
                    initializer=tf.random_normal_initializer(0, 0.3)
                )
            b3 = tf.get_variable(
                    "b3", 
                    shape=(1, hidden_unit_list2[2]), 
                    initializer=tf.constant_initializer(0.1)
                )
            output = tf.matmul(layer2, W3) + b3

    return output

with tf.variable_scope("loss"):
    loss = tf.reduce_mean(cross_entropy, name = "loss")
with tf.variable_scope("accutacy"):

with tf.variable_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

data_x, data_y = Load_Data()
data_index = 1
Matrix1 = Select_Data_Mat(data_x,data_y,data_index,D)