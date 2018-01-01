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
def create_one_hot_dict(file_item, num_item = NUM_ITEM):
    """
        input:
            file_item: file chứa dữ liệu item items.txt
        return: 
            dictionary[item_string] = one_hot_vector (np.array())
    """
    item_index = 0
    dict_item = {}
    with open(file_item, "r") as file:
        print("begin read file and create dict one hot")
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

def convert_item_to_one_hot(items, dict_items):
    one_hot_items = []
    for item in items:
        one_hot_items.append(dict_items[item])
    one_hot_items = np.array(one_hot_items)
    # print(one_hot_items.shape)
    return one_hot_items

#tao input dau vao
def create_input_a_output(file_name):
    """
        file_name: file chua cac session train.txt hoac test.txt
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
                session.append(item)
            model_input.append(session[0:-1])
            model_output.append(session[1:])
    print("Finish create input and output.")
    return np.array(model_input),np.array(model_output)

# dictionary = create_one_hot(FILE_ITEM)
# input, output = create_input_a_output(FILE_TEST, dictionary)

def shuffle_data(data_input, data_target):
    """Shuffle data"""
    print("Shuffle the data")
    # print(data_input[0])
    # print(data_target[0])
    shuffled_ix = np.random.permutation(np.arange(len(data_target)))
    data_input_shuffle = data_input[shuffled_ix]
    data_target_shuffle = data_target[shuffled_ix]
    return data_input_shuffle, data_target_shuffle
#build model

# Parameters
learning_rate = 0.01
epochs = 2
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

x = tf.placeholder(tf.float32, [batch_size, None, n_items], name = "X")
y = tf.placeholder(tf.float32, [batch_size, None, n_items], name = "Y")
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
    return tf.sigmoid(tf.matmul(outputs, weight["out"]) + bias["out"])
pred = dynamic_rnn_pred(x, weight, bias, seqlen)
pred = tf.reshape(pred, [batch_size, seq_max_len, n_items])
print(pred)

def loss_function(pred, target, length):
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

def calc_recall_20(pred, target, seqlen):
    top_items = tf.nn.top_k(pred, 20).indices
    top_items = top_items[0]
    desire_index = tf.cast( tf.argmax(target,axis = 2), tf.int32) 
    desire_index = desire_index[0]
    accuracy = 0.0
    for i in range(seq_max_len):
        top_k_i = top_items[i]
        accuracy += tf.cast(tf.equal(tf.reduce_sum(tf.cast(tf.equal(top_k_i,desire_index[i]),tf.float32)),1), tf.float32) 
    accuracy /= tf.cast(seqlen,tf.float32)
    # tf.nn.in_top_k(top_items, desire_index, 20)
    # desire_index = tf.reshape(desire_index, [-1, n_hidden])
    # print(top_items)
    # print(desire_index)
    return accuracy

recall20 = calc_recall_20(pred, y, seqlen)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    #get input data and output data
    input_set, output_set = create_input_a_output(FILE_TEST)
    dict_items = create_one_hot_dict(FILE_ITEM)
    for _ in range(epochs):
        #shuffle data
        input_set, output_set = shuffle_data(input_set, output_set)
        for i in range(len( input_set)):
            # get curr data for input, output
            curr_input = input_set[i]
            curr_output = output_set[i]
            
            # convert to one-hot encoding
            curr_input = np.array([convert_item_to_one_hot(curr_input, dict_items)]) 
            curr_output = np.array([ convert_item_to_one_hot(curr_output, dict_items)]) 

            # push one-hot encoding to model
            _, cost, acc = sess.run([optimizer, loss, recall20], feed_dict={x: curr_input, y:curr_output})
            print("Accuracy: " + str(acc))
            print("Cost: " + str(cost))
            print("\n\n")
            # evaluate model      
