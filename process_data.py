from collections import deque

PATH_TO_TRAIN = "../../Session-based with RNN data/data/rsc15_train_full.txt"
PATH_TO_TEST = "../../Session-based with RNN data/data/rsc15_test.txt"
PATH_TO_LIST_ITEMS = "../../Session-based with RNN data/data/rsc15_list_items.txt"
PATH_TO_LIST_ITEMS_PER_SESSION= "../../Session-based with RNN data/data/rsc15_list_items_per_session.txt"
PATH_TO_LIST_ITEMS_TEST = "../../Session-based with RNN data/data/rsc15_list_items_test.txt"
PATH_TO_LIST_ITEMS_PER_SESSION_TEST = "../../Session-based with RNN data/data/rsc15_list_items_per_session_test.txt"
def build_data_for_list_items_and_list_items_of_sessions(path_to_train_file):
    count = 0
    num_items = 0
    session_id_before = ""
    list_items_of_sessions = {} #{ss1:[],ss2:[], ...}
    list_items_per_session = [] #[[],[],[], ...]
    list_items = []
    set_items = {}
    with open(path_to_train_file,"r") as file:
        file.readline()
        for line in file:
            count += 1
            print(count)
            input_str = line.split("\t")
            session_id, item_id = input_str[0],input_str[1]
            list_items.append(item_id)
            if session_id == session_id_before:
                list_items_of_sessions[session_id].append(item_id)
            else:
                if session_id_before != "":
                    list_items_per_session.append(list_items_of_sessions[session_id_before])
                list_items_of_sessions[session_id] = [item_id]
                session_id_before = session_id
    # print(len(list_items_of_sessions)) (number of session)
    # print(count)    (number of event)
    set_items = set(list_items)
    # print(len(set_items)) )nnumber of items)
    return list_items_per_session,set_items 

def write_list_items_to_file(list_items, path_to_list_items_file):
    with open(path_to_list_items_file,"w") as file:
        for item in list_items:
            file.write(item+"\n")
        file.close()
    return

def write_list_items_per_session(list_items_per_session, path_to_list_items_per_session_file):
    with open(path_to_list_items_per_session_file,"w") as file:
        for list_items in list_items_per_session:
            for item in list_items:
                file.write(item)
                file.write("\t")
            file.write("\n")

def build_minibatch_file(batch_size, minibatch_input_file, minibatch_output_file, path_to_list_items_per_session_file):
    data = []
    list_items_per_session_for_minibatch = []
    new_list_items = []
    num_batch = 0   
    minibatch_input = []
    minibatch_output = []
    # nap du lieu vao list data
    with open(path_to_list_items_per_session_file,"r") as file: 
        for line in file: 
            items_str = line.split("\t")
            del items_str[-1]
            data.append(items_str)
    data = deque(data)
    # khoi tao du lieu ban dau
    for _ in range(batch_size):
        new_list_items = data.popleft()
        list_items_per_session_for_minibatch.append(new_list_items)
    
    while True:
        #check do dai data 
        print(len(data))
        
        # check list co length nho hon 2, thay bang list moi
        for i in range(batch_size):
            if len(list_items_per_session_for_minibatch[i]) < 2:
                if len(data) == 0:
                    break
                new_list_items = data.popleft()
                list_items_per_session_for_minibatch[i] = new_list_items
        if len(data) == 0:
            break 
        one_batch_input = []
        one_batch_output = []
        for list_items in list_items_per_session_for_minibatch:
            # print(list_items)
            one_batch_input.append(list_items.pop(0))
            one_batch_output.append(list_items[0])
        num_batch += 1
        print("numbatch :"+str(num_batch))
        minibatch_input.append(one_batch_input)
        minibatch_output.append(one_batch_output)

    # ghi ra file 
    with open(minibatch_input_file,"w") as file:
        count = 0
        for batch in minibatch_input:
            for item in batch:
                file.write(item)
                file.write("\t")
            file.write("\n")
            print(count)
            count+=1
    with open(minibatch_output_file,"w") as file:
        count = 0
        for batch in minibatch_output:
            for item in batch:
                file.write(item)
                file.write("\t")
            file.write("\n")
            print(count)
            count+=1
    return 
list_items_per_session, set_items = build_data_for_list_items_and_list_items_of_sessions(PATH_TO_TRAIN)
write_list_items_to_file(set_items, PATH_TO_LIST_ITEMS)
write_list_items_per_session(list_items_per_session, PATH_TO_LIST_ITEMS_PER_SESSION)
build_minibatch_file(5,"../../Session-based with RNN data/train_data/input.txt","../../Session-based with RNN data/train_data/output.txt",PATH_TO_LIST_ITEMS_PER_SESSION)

list_items_per_session, set_items = build_data_for_list_items_and_list_items_of_sessions(PATH_TO_TEST)
write_list_items_to_file(set_items, PATH_TO_LIST_ITEMS_TEST)
write_list_items_per_session(list_items_per_session, PATH_TO_LIST_ITEMS_PER_SESSION_TEST)
build_minibatch_file(5,"../../Session-based with RNN data/train_data/input_test.txt","../../Session-based with RNN data/train_data/output_test.txt",PATH_TO_LIST_ITEMS_PER_SESSION_TEST)