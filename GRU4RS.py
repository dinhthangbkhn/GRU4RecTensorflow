"""
created by Thang Dinh at Dec 11, 2017
RNN for session based recommend system 
"""

import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
# set up parameter
INPUT_FILE = "../../Session-based with RNN data/train_data/input.txt"
OUTPUT_FILE = "../../Session-based with RNN data/train_data/output.txt"
INPUT_FILE_TEST ="../../Session-based with RNN data/train_data/input_test.txt"
OUTPUT_FILE_TEST = "../../Session-based with RNN data/train_data/output_test.txt"
PATH_TO_LIST_ITEMS = "../../Session-based with RNN data/data/rsc15_list_items.txt"
BATCH_SIZE = 5
rnn_size = 10 #so luong GRU 
epochs_num = 20
number_hidden_unit = 50 #number of unit in hidden layer
dropout_keep_prob = 0.2
embedding_size = 50
learning_rate = 0.05

# load data from file 

def get_data_from_file(file_name):
    data = []
    print("Get data from file "+file_name)
    with open(file_name,"r") as file:
        for line in file:
            input_unit = line.split("\t")
            del input_unit[-1]
            input_unit = list(map(int, input_unit))
            data.append(input_unit)
    return data


#build the dictionary: item -> index, return dictionary, number of item
def indexing_list_items(file_name):
    print("Indexing list items from "+file_name)
    items_index = {}
    count = 0
    with open(file_name, "r") as file:
        for line in file:
            line = line[:-1]
            items_index[int(line)] = count
            count += 1
    return items_index, count

#indexing the data
def indexing_train_data(item_to_index, list_of_list):
    print("Indexing train data based on item to index")
    for i in range(len(list_of_list)):
        for j in range(len(list_of_list[i])):
            list_of_list[i][j] = item_to_index[list_of_list[i][j]]




#create one hot encoding minibatch
def create_one_hot_for_input(minibatch_input_items, number_items):
    print("Create one-hot vector")
    one_hot_minibatch_input = []
    for input_item in minibatch_input_items:
        one_hot_vector = np.zeros(number_items)
        one_hot_vector[input_item] = 1
        one_hot_list_for_rnn = [] #do moi dau vao can rnn_size input giong nhau
        for _ in range(rnn_size):
            one_hot_list_for_rnn.append(one_hot_vector)
        one_hot_minibatch_input.append(one_hot_list_for_rnn)
    one_hot_minibatch_input = np.array(one_hot_minibatch_input)
    return one_hot_minibatch_input

def create_one_hot_for_output(minibatch_input_items, number_items):
    one_hot_minibatch_input = []
    for input_item in minibatch_input_items:
        one_hot_vector = np.zeros(number_items)
        one_hot_vector[input_item] = 1
        one_hot_minibatch_input.append(one_hot_vector)
    one_hot_minibatch_input = np.array(one_hot_minibatch_input)
    return one_hot_minibatch_input

def shuffle_data(data_input, data_target):
    print("Shuffle the data")
    # print(data_input[0])
    # print(data_target[0])
    shuffled_ix = np.random.permutation(np.arange(len(data_target)))
    data_input_shuffle = data_input[shuffled_ix]
    data_target_shuffle = data_target[shuffled_ix]
    return data_input_shuffle, data_target_shuffle

item_to_index, number_items = indexing_list_items(PATH_TO_LIST_ITEMS)
# build model 

x_data = tf.placeholder(tf.float32, [BATCH_SIZE, rnn_size, number_items],name="x_data")
y_output = tf.placeholder(tf.float32, [BATCH_SIZE, number_items], name="y_output")

x_data_reshape = tf.reshape(x_data, [-1, number_items])
x_data_split = tf.split(x_data_reshape,num_or_size_splits=rnn_size,axis=0) #slit data for each rnn cell

cell = tf.nn.rnn_cell.GRUCell(num_units = number_hidden_unit)
output, state = tf.nn.static_rnn(cell,inputs = x_data_split, dtype=tf.float32)
output_drop = tf.nn.dropout(output, dropout_keep_prob)
output_last = tf.gather(output_drop, output_drop.get_shape()[0] - 1)
weight = tf.Variable(tf.truncated_normal([number_hidden_unit, number_items], stddev = 0.1),name="weight")
bias = tf.Variable(tf.constant(0.1, shape=[number_items]),name="bias")

pred = tf.matmul(output_last, weight)+bias

#calculate the loss function 
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y_output))
desired_index = tf.cast(tf.argmax(y_output,1),tf.int32)

#optimization
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

# evaluate model
top_items = tf.nn.top_k(pred, 20).indices
print(top_items)
accuracy = 0.0
for i in range(BATCH_SIZE):
    row = tf.gather(top_items,i)
    accuracy+=tf.cast(tf.equal(tf.reduce_sum(tf.cast(tf.equal(row,desired_index[i]),tf.float32)),1), tf.float32) 
accuracy=accuracy/BATCH_SIZE

init = tf.global_variables_initializer()

# training begin
with tf.Session() as sess:
    sess.run(init)

    # data_input_train = np.array(get_data_from_file(INPUT_FILE))
    # data_target_train = np.array(get_data_from_file(OUTPUT_FILE))
    data_input_test = np.array(get_data_from_file(INPUT_FILE_TEST))
    data_target_test = np.array(get_data_from_file(OUTPUT_FILE_TEST))
    print(data_input_test[0])
    print(data_target_test[0])

    # indexing_train_data(item_to_index, data_input_train)
    # indexing_train_data(item_to_index, data_target_train)
    indexing_train_data(item_to_index, data_input_test)
    indexing_train_data(item_to_index, data_target_test)
   
    for _ in range(epochs_num):
        data_input, data_target = shuffle_data(data_input_test, data_target_test)       
        # numbatches = int(len(data_input_test)/BATCH_SIZE)
        print("Length of data: " + str(len(data_target)))
        for i in range(BATCH_SIZE):
            x_input = data_input[i:i+1]
            y_output = data_target[i:i+1]
            x_input = x_input.transpose() #doi chieu (1,batch_size) -> (batch_size,1)
            x_input = create_one_hot_for_input(x_input, number_items)
            y_output = y_output.transpose()
            y_output = create_one_hot_for_output(y_output, number_items)
            x_input = x_input.astype(np.float32)
            y_output = y_output.astype(np.float32)
            print(y_output.shape)
            _, cost, acc = sess.run([optimizer, loss, accuracy], feed_dict={x_data: x_input, y_output: y_output})
            print("Accuracy: "+str(acc))
            print("Cost: "+str(cost))
            if i > 5:
                break
        break
