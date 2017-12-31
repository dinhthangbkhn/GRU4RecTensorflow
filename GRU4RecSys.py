"""
created by Thang Dinh at Dec 11, 2017
RNN for session based recommend system 
"""

import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 

FILE_TRAIN = "../data/gr2-sessions/train.txt"
FILE_TEST = "../data/gr2-sessions/test.txt"
FILE_ITEM = "../data/gr2-sessions/items.txt"
NUM_ITEM = 37483

#create one hot vector for each item
def create_one_hot(file_item, num_item = NUM_ITEM):
    """
        input:
            file_item: file chứa dữ liệu item items.txt
        return: 
            dictionary[item_string] = one_hot_vector (np.array())
    """
    item_index = 0
    dict_item = {}
    with open(file_item, "r") as file:
        print("begin read file and create vector one hot")
        for line in file:
            item_str = line 
            item_one_hot = np.zeros(num_item)
            item_one_hot[item_index] = 1.0 
            dict_item[item_str[0:-1]] = item_one_hot
            item_index += 1
            if item_index%2000 == 0:
                print("Indexing: " + str(item_index))
        print("end read file and create vector one hot, \nNumber vector: " + str(item_index+1))
    return dict_item

#tao input dau vao
def create_input_a_output(file_name, dict_item):
    """
        file_name: file chua cac session train.txt hoac test.txt
        dict_item: chuyen tu str --> numpy array
        output: 
            input cua model
            output cua model
    """
    model_input = []
    model_output = []
    print("Open file and create input, output ...")
    with open(file_name, "r") as file:
        for line in file:
            session = []
            items_str = line.split("\t")[0:-1]
            for item in items_str:
                session.append(dict_item[item])
            model_input.append(session[0:-1])
            model_output.append(session[1:])
    print("Finish create input and output.")
    return model_input, model_output

# dictionary = create_one_hot(FILE_ITEM)
# input, output = create_input_a_output(FILE_TEST, dictionary)

#build model

# Parameters
learning_rate = 0.01
epochs = 10000
batch_size = 128
display_step = 200

# Network Parameters
seq_max_len = 30 # Sequence max length
n_hidden = 64 # hidden layer num of features
n_items = 37843 #number of items
batch_size = 1

weight = {
    "out": tf.Variable(tf.truncated_normal([n_hidden, n_items], stddev = 0.1),name="weight")
}
bias = {
    "out": tf.Variable(tf.constant(0.1, shape=[n_items]),name="bias")
}
def length_of_sequence(sequence):
    """ sequence: [batchsize, number node, number items]"""
    used = tf.sign(tf.reduce_max( tf.abs(sequence), axis = 2))
    sum = tf.reduce_sum(used, axis = 1)
    length = tf.cast(sum, tf.int32)
    return length

x = tf.placeholder(tf.float32, [batch_size, seq_max_len, n_items], name = "X")
y = tf.placeholder(tf.float32, [batch_size, seq_max_len, n_items], name = "Y")
seqlen = length_of_sequence(x)
def dynamic_rnn_pred(x, weight, bias, seqlen):
    # create 2 LSTMCells
    rnn_layers = [tf.nn.rnn_cell.GRUCell(size) for size in [n_hidden]]

    # create a RNN cell composed sequentially of a number of RNNCells
    multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

    # 'outputs' is a tensor of shape [batch_size, max_time, n_hidden]
    # 'state' is a N-tuple where N is the number of LSTMCells containing a
    # tf.contrib.rnn.LSTMStateTuple for each cell
    outputs, state = tf.nn.dynamic_rnn(cell=multi_rnn_cell, inputs=x, dtype=tf.float32, sequence_length=seqlen)
    # outputs = tf.stack(outputs)
    outputs = tf.reshape(outputs, [-1, n_hidden])
    # outputs = outputs[0]
    # print(outputs)
    return tf.matmul(outputs, weight["out"]) + bias["out"]
pred = dynamic_rnn_pred(x, weight, bias, seqlen)
print(pred)

def loss_function(pred, target, length):
    pred = tf.reshape(pred, [batch_size, seq_max_len, n_items])
    # print(pred)
    cross_entronpy = target * tf.log(pred)
    # print(cross_entronpy)
    cross_entronpy = -tf.reduce_sum(cross_entronpy, 2)
    # print(cross_entronpy)
    mask = tf.sign(tf.reduce_max( tf.abs(target), axis = 2))
    # print(mask)
    cross_entronpy *= mask
    # print(cross_entronpy)
    cross_entronpy = tf.reduce_sum(cross_entronpy, 1)
    # print(cross_entronpy)
    cross_entronpy /= tf.reduce_sum(mask, 1)
    # print(cross_entronpy)
    return cross_entronpy 
loss = loss_function(pred, y, seqlen)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

def Calc_recall_20(pred, target):
    