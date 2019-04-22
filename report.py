from matplotlib import pyplot as plt
import numpy as np
import csv 

def load_run_viz(dataset_name):
    data = np.empty((0, 3))
    i = 0
    with open(dataset_name, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        for train_error, val_error in reader:
            if train_error == "train-error":
                continue
            train_error = float(train_error)
            val_error = float(val_error)

            #           [index, train_error, val_error]
            i += 1
            data = np.append(data, np.array([[i, train_error, val_error]]), axis=0)

    plt.plot(data[:,0], data[:,1], c="black")
    plt.plot(data[:,0], data[:,2], c="yellow")
    plt.show()

load_run_viz("res/big-pics/metrics.txt")
load_run_viz("res/lot-of-pics/metrics.txt")