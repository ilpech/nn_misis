# %load report.py
from matplotlib import pyplot as plt, axes
import os
import numpy as np
import csv 

def clear():
    plt.clf()
    plt.cla()
    plt.close()

dict_path = os.path.join(os.getcwd(), "res/lot-of-pics/gel_cls.003_classes.txt")
test = np.genfromtxt("testing/test_error_matrix.csv", delimiter=',')
val = np.genfromtxt("testing/val_error_matrix.csv", delimiter=',')
with open(dict_path) as f:
    content = f.readlines()
class_names = [x.strip() for x in content]

def heatmap(data):
    plt.xticks(np.arange(20), class_names)
    plt.yticks(np.arange(20), class_names)
    plt.xticks(rotation=90)
    plt.imshow(data, cmap='hot', interpolation='nearest')
    plt.show()
    clear()

def metrics(dataset_name, description = ""):
    data = np.empty((0, 3))
    with open(dataset_name, 'r') as csvfile:
        i = 0
        reader = csv.reader(csvfile, delimiter=' ')
        for train_error, val_error in reader:
            if train_error == "train-error":
                continue
            train_error = float(train_error)
            val_error = float(val_error)

        #                   [index, train_error, val_error]
            i += 1
            data = np.append(data, np.array([[i, train_error, val_error]]), axis=0)
    plt.title("Метрики обучения: " + description)
    plt.xticks(rotation=90)
    plt.plot( data[:,1], label='Train error')
    plt.plot( data[:,2], label='Validation error')
    plt.ylim([0,1])
    plt.legend()
    plt.show()
    clear()

def errorsTest():
    plt.title("Матрица ошибок на тестовых данных")
    heatmap(test)

def errorsTestVal():
    plt.title("Матрица ошибок на тестовых и валидационных данных")
    heatmap(test + val)

def errors():
    prefixes = ["test", "val"]
    suffixes = ["fn", "fp", "tp"]
    for prefix in prefixes: 
        for suffix in suffixes:
            print(prefix + "(" + suffix + "): ")
            filename = os.path.join(os.getcwd(), "testing", prefix + "_" + suffix + ".csv")
            metric = np.genfromtxt(filename, delimiter=',')
            plt.title("Ошибки: " + prefix + " " + suffix)
            plt.bar(x=class_names, height=metric, label=class_names)
            plt.xticks(rotation=90)
            plt.show()
            print(metric)

errorsTest()
errorsTestVal()
errors()

metrics("res/big-pics/metrics.txt", "версия сети с большими изображениями ")
metrics("res/lot-of-pics/metrics.txt", "версия сети с фрагментами оригальных изображений")