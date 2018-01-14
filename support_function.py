
#open file rsc_15_test
def session_filter(file_in, seqlen, file_out):
    """clear session that has length large than seqlen"""
    with open(file_in,"r") as file:
        new_file_session = []
        for line in file:
            sequence = line.split("\t")
            if len(sequence) > 20:
                continue
            new_file_session.append(line)
    with open(file_out, 'w') as file:
        for line in new_file_session:
            file.write(line)
    
# session_filter("../data/gr2-sessions/test.txt", 20, "../data/gr2-sessions/new_test_20.txt")

