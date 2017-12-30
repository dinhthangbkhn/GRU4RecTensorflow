# from collections import deque
PATH_TO_TRAIN_GR2RSC15 = "../data/gr2-rsc15/rsc15_train_full.txt"
PATH_TO_TEST_GR2RSC15 = "../data/gr2-rsc15/rsc15_test.txt"
PATH_TO_TRAIN_SESSION = "../data/gr2-sessions/train.txt"
PATH_TO_TEST_SESSION = "../data/gr2-sessions/test.txt"
PATH_TO_LIST_ITEM = "../data/gr2-sessions/items.txt"
def create_sessions(file_name):
    """
        file_name: file rsc15_train_full hoặc rsc15_test
        create sessions
    """
    with open(file_name, "r") as file:
        file.readline()
        pre_sess_id = ""
        num_sess = 0
        sessions = []
        curr_sess_items = []
        for line in file:
            curr_line_list = line.split("\t")
            curr_sess_id, curr_item = curr_line_list[0], curr_line_list[1]
            if curr_sess_id == pre_sess_id:
                curr_sess_items.append(curr_item)
            else:
                sessions.append(curr_sess_items)
                num_sess += 1
                if num_sess%500 == 0:
                    print(num_sess)
                curr_sess_items = [curr_item]
                pre_sess_id = curr_sess_id
        sessions.append(curr_sess_items)
    return num_sess, sessions

def write_sessions_to_file(file_target, sessions):
    """Write session that created to file"""
    with open(file_target, "w") as file:
        for session in sessions:
            for item in session:
                file.write(item+"\t")
            file.write("\n")
    return

def write_list_item_to_file(file_input, file_target):
    """Collect the item and write to file"""
    list_item = []
    print("Reading file input ...")
    with open(file_input, "r") as file:
        file.readline()
        for line in file:
            curr_line_list = line.split("\t")
            list_item.append(curr_line_list[1])
    set_item = set(list_item)
    print("Finish reading file input, writing to file target...")
    with open(file_target, "w") as file:
        for item in set_item:
            file.write(item)
            file.write("\n")
    print("Finish write list item to file")
    return

def run_process_gr2_rsc15():
    """Run this function to process data"""
    num_sess, sessions = create_sessions(PATH_TO_TEST_GR2RSC15)
    write_sessions_to_file(PATH_TO_TEST_SESSION, sessions)
    num_sess, sessions = create_sessions(PATH_TO_TRAIN_GR2RSC15)
    write_sessions_to_file(PATH_TO_TRAIN_SESSION, sessions)
    write_list_item_to_file(PATH_TO_TRAIN_GR2RSC15, PATH_TO_LIST_ITEM)
    return
