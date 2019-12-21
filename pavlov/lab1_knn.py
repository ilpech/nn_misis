import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle as sklearn_shuffle
import pandas as pd
import sys

def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

class KNN:
    @staticmethod
    def euclidean_distance(row1, row2):
        distance = 0.0
        for i in range(len(row1)-1):
            distance += (row1[i] - row2[i])**2
        return sqrt(distance)

    @staticmethod 
    def manhettan_dist(row1, row2):
        distance = 0.0
        for i in range(len(row1)-1):
            distance += (row1[i] - row2[i])
        return abs(distance)
        
    @staticmethod
    def get_neighbors(train, test_row, num_neighbors, dist_metric=euclidean_distance):
        distances = list()
        for train_row in train:
            dist = dist_metric(test_row, train_row)
            distances.append((train_row, dist))
        distances.sort(key=lambda tup: tup[1])
        neighbors = list()
        for i in range(num_neighbors):
            neighbors.append(distances[i][0])
        return neighbors
    
    @staticmethod
    def predict_classification(train, test_row, num_neighbors, dist_metric):
        neighbors = KNN.get_neighbors(train, test_row, num_neighbors, dist_metric)
        output_values = [row[-1] for row in neighbors]
        prediction = max(set(output_values), key=output_values.count)
        return prediction

    @staticmethod
    def k_nearest_neighbors(train, test, num_neighbors, dist_metric):
        predictions = []
        for row in test:
            output = KNN.predict_classification(train, row, num_neighbors, dist_metric)
            predictions.append(output)
        return(predictions)

    @staticmethod 
    def evaluate_knn(data, dist_metric, num_neighbors, test_d_perc=0.2):
        train_data = data[:int(1-test_d_perc*len(data))]
        test_data = data[int(1-test_d_perc*len(data)):]
        predict = KNN.k_nearest_neighbors(train_data, test_data, num_neighbors, dist_metric)
        gt = [row[-1] for row in test_data]
        return accuracy_metric(gt, predict)
       
def dataset_minmax(dataset):
    minmax = []
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax

def find_same_columns(dataset):
    cols2drop = []
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        if all(x==col_values[0] for x in col_values):
            cols2drop.append(i)
    return cols2drop
    
def normalize_dataset(dataset):
    minmax = dataset_minmax(dataset)
    for row in dataset:
        for i in range(len(row)):
            if(isinstance(row[i], str)):
                break
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

sys.path.append('../..')
from project_tools import *
data_p = os.path.join(os.getcwd(), 'data', 'ecoli.data')
df = pd.read_csv(data_p, header=None, delim_whitespace=True, usecols=[2,3,4,5,6,7,8])
n_samples = len(df)
print(n_samples)
df = sklearn_shuffle(df)
dataset = np.array(df)
normalize_dataset(dataset)
find_same_columns(dataset)
max_eucl = 0.0
max_eucl_k = -1
max_manh = 0.0
max_manh_k = -1
for k_size in range(1,n_samples-1):
    try:
        acc_euclid = KNN.evaluate_knn(dataset, KNN.euclidean_distance, k_size, 0.1)
        acc_manh = KNN.evaluate_knn(dataset, KNN.manhettan_dist, k_size, 0.1)
        print('KNN ACC EUCLID: {} | KNN ACC MANHETTAN: {} | k_size = {}'.format(acc_euclid, acc_manh, k_size))
    except IndexError:
        pass
    if(acc_euclid > max_eucl):
        max_eucl = acc_euclid
        max_eucl_k = k_size
    if(acc_manh > max_manh):
        max_manh = acc_manh
        max_manh_k = k_size
print('max acc with euclid metric = {} with k_size = {}'.format(max_eucl, max_eucl_k))
print('max acc with manh metric = {} with k_size = {}'.format(max_manh, max_manh_k))

# KNN.evaluate_knn(dataset, KNN.euclidean_distance, 3)





