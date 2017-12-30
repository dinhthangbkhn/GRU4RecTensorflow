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
            file_item: file chứa dữ liệu item 
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
            if item_index%1000 == 0:
                print("Index: " + str(item_index))
        print("end read file and create vector one hot, \nNumber vector: " + str(item_index+1))
    return dict_item
