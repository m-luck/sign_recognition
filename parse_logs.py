import argparse
import sys
from matplotlib import pyplot as plt

def parse(filename):
    training_count = 0
    training_avg_loss = 0
    training_avg_plot = []
    val_plot = []
    val_acc = []
    with open(filename, "r") as f:
        for line in f:
            if is_training_line(line):
                training_count += 1
                training_avg_loss = (get_training_loss(line) + training_avg_loss * (training_count - 1)) / training_count
            if is_val_line(line):
                training_avg_plot.append(training_avg_loss)
                val_plot.append(get_val_loss(line))
                val_acc.append(get_val_acc(line))
                training_avg_loss = 0
                training_count = 0
        print(training_avg_plot, val_plot)
        return training_avg_plot, val_plot, val_acc
                
            
            
def is_training_line(line):
    if line[0:2] == "Tr":
        return True
    return False

def is_val_line(line):
    if line[0:2] == "Va":
        return True
    return False

def get_training_loss(line):
    print(line.split("Loss: "))
    return float(line.split("Loss: ")[1])

def get_val_loss(line):
    return float(line.split("loss: ")[1].split(",")[0])

def get_val_acc(line):
    return float(line.split("(")[1].split("%")[0])

train, val, val_acc = parse(sys.argv[1])

tr, = plt.plot(train, label="train")
va, = plt.plot(val, label="val")
plt.legend(handles=[tr, va])
plt.xlabel("epoch")
plt.ylabel("loss")

# vac, = plt.plot(val_acc, label="val acc")
# plt.legend(handles=[vac])
# plt.xlabel("epoch")
# plt.ylabel("percentage acc")

plt.show()