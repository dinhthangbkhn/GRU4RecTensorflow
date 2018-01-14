import tensorflow as tf 
import numpy as np 
import win_unicode_console
win_unicode_console.enable()

FILE_TRAIN = "../data/gr2-sessions/train.txt"
FILE_TEST = "../data/gr2-sessions/new_test_20.txt"
FILE_ITEM = "../data/gr2-sessions/items.txt"

def create_item_to_id(file_item):
    """ file_item: file chua id tren web cua cac item"""
    with open(file_item, "r") as file:
        items = file.readlines()
        for i in range(len(items)):
            items[i] = int(items[i][0:-1]) 
        # print(items)
    item_to_id = dict(zip(items, range(len(items))))
    # print(item_to_id)
    return item_to_id, len(item_to_id)

def create_data(file_input):
    item_to_id, n_items = create_item_to_id(FILE_ITEM)
    data_input = []
    data_target = []
    with open(file_input, "r") as file:
        for line in file:
            items = [item_to_id[int(x)] for x in line.split("\t")[0:-1] ] 
            data_input.append(items[0:-1])
            data_target.append(items[1:])
    return np.array(data_input), np.array(data_target), n_items

def convert_session_to_onehot(session, n_items):
    session_onehot = []
    for item in session:
        onehot = np.zeros(n_items)
        onehot[item] = 1.0
        session_onehot.append(onehot)
    return np.array(session_onehot)

def shuffle_data(data_input, data_target):
    """Shuffle data"""
    print("Shuffle the data ..")
    shuffled_ix = np.random.permutation(np.arange(len(data_target)))
    data_input_shuffle = data_input[shuffled_ix]
    data_target_shuffle = data_target[shuffled_ix]
    print("End shuffle the data")
    return data_input_shuffle, data_target_shuffle

# shuffle_data(input, output)

#SET UP PARAMETER
n_hidden = 10
batch_size = 1
n_items = 6751
seq_max_len = 80
init_scale = 0.05
learning_rate = 0.01
epochs = 2

x = tf.placeholder(tf.int32, [batch_size, None])
y = tf.placeholder(tf.int32, [batch_size, None])
x_length = tf.placeholder(tf.int32)
embedding = tf.Variable(tf.random_uniform([n_items, n_hidden], -init_scale, init_scale))
inputs = tf.nn.embedding_lookup(embedding, x)
print(inputs)
# rnn_layers = [tf.nn.rnn_cell.GRUCell(size) for size in [n_hidden]]
# multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
cell = tf.contrib.rnn.GRUCell(n_hidden)
outputs_rnn, state = tf.nn.dynamic_rnn(cell, inputs=inputs, dtype=tf.float32)#, sequence_length = x_length)
outputs = tf.reshape(outputs_rnn, [-1, n_hidden])

softmax_w = tf.Variable(tf.random_uniform([n_hidden, n_items], -init_scale, init_scale))
softmax_b = tf.Variable(tf.random_uniform([n_items], -init_scale, init_scale))
logits = tf.nn.xw_plus_b(outputs, softmax_w, softmax_b)
logits = tf.reshape(logits, [batch_size, -1, n_items])
print(logits)
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits,labels =  y)

# logits = tf.reshape(logits, [batch_size, -1, n_items])
# print(logits)
# loss = tf.contrib.seq2seq.sequence_loss(logits, y, tf.ones([batch_size, seq_max_len], dtype=tf.float32), average_across_timesteps=False, average_across_batch=True)
# loss = tf.reduce_mean(loss)

optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
top20 = tf.nn.top_k(logits, 20)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    data_input, data_output, nitems = create_data(FILE_TEST)
    for _ in range(epochs):
        data_input, data_output = shuffle_data(data_input, data_output)
        for i in range(len(data_input)):
            curr_input = np.array([data_input[i]])
            curr_output = np.array([data_output[i]])

            print(curr_input.shape)
            # print(len(curr_input[1]))
            print(curr_output.shape)
            output_rnn_value, loss_value = sess.run([outputs_rnn, loss], feed_dict = {x: curr_input, y: curr_output, x_length: len(curr_input[0])})
            print(output_rnn_value)
            print(loss_value)
            break
