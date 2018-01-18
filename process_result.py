from matplotlib import pyplot as plt

def draw_loss_function(file_name, g_name):
    list = []
    with open(file_name,"r") as file:
        for line in file:
            if line != "Loss\n":
                number = float(line[0:-1])
                if number > 0.05:
                    list.append(number)
                
    plt.plot(list)
    plt.title(g_name)
    plt.ylabel("loss")
    plt.xlabel("step * 1000")
    plt.show()
draw_loss_function("loss500.txt", "Loss with hidden units = 500")           
draw_loss_function("loss100.txt", "Loss with hidden units = 100")
draw_loss_function("loss10.txt", "Loss with hidden units = 10")

def draw_recall_function(file_name, g_name, gy_label):
    list = []
    with open(file_name, "r") as file:
        recall = file.readlines()
        recall = recall[0].split(".")
        for val in recall:
            print(val[0:-1])
            number = float("0."+ val[0:-1])
            if number > 0.05:
                list.append(number)
        # print(recall)
        # print(list)
    plt.plot(list)
    plt.title(g_name)
    plt.ylabel(gy_label)
    plt.xlabel("step * 1000")
    plt.show()
# draw_recall_function("resutl_hidden10_rate0_01.txt","Recall@20, hidden units = 10", "Recall@20")
# draw_recall_function("resutl_hidden100_rate0_01.txt","Recall@20, hidden units = 100", "Recall@20")
# draw_recall_function("resutl_hidden500_rate0_01.txt","Recall@20, hidden units = 500", "Recall@20")