import argparse
import sys
from matplotlib import pyplot as plt

def parse(filename):
    training_count = 0
    training_avg_loss = 0
    training_avg_plot = []
    val_plot = []
    with open(filename, "r") as f:
        for line in f:
            if is_training_line(line):
                training_count += 1
                training_avg_loss = (get_training_loss(line) + training_avg_loss * (training_count - 1)) / training_count
            if is_val_line(line):
                training_avg_plot.append(training_avg_loss)
                val_plot.append(get_val_loss(line))
                training_avg_loss = 0
                training_count = 0
        print(training_avg_plot, val_plot)
        return training_avg_plot, val_plot
                
            
            
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

train, val = parse(sys.argv[1])

plt.plot(train)
plt.plot(val)

plt.show()