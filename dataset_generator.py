import matplotlib.pyplot as plt
import sys
import os
import numpy as np
import math

import time

def euclidean_distance(first_point, second_point):
    if first_point.shape != second_point.shape:
        print("first_point.shape != second_point.shape")
        return
    sum = 0.0
    for dim in range(len(first_point)):
        sum += (first_point[dim] - second_point[dim])**2
    return math.sqrt(sum)

def make_dataset(n_features = 2, n_samples = 100000, n_classes = 40,
                 linear_separable=True, n_clusters=1, max_intersection_percentage = 0.9, draw=True):
    generator = np.random.RandomState(420)
    center_box = (0, 100.0)
    cluster_std = 1.0
    classes = generator.uniform(center_box[0], center_box[1], size=(n_classes, n_features))
    n_samples_per_class = [int(n_samples // n_classes)] * n_classes
    for i in range(n_samples % n_classes):
        n_samples_per_class[i] += 1
    cluster_std = np.full(len(classes), cluster_std)
    X = np.zeros((0,n_features))
    y = np.zeros((0))
    for i, (n, std) in enumerate(zip(n_samples_per_class, cluster_std)):
        if(i == 0):
            X = np.append(X, generator.normal(loc=classes[i], scale=std,
                                      size=(n, n_features)), axis=0)
            y = np.append(y, np.full(n,i), axis=0)
        else:
            # generation mode for non linear sep case
            if not linear_separable:
                n_cluster = n_classes / n_clusters
                if i % n_cluster == 0:
                    shift = generator.uniform(-10.0, 10.0, size=(1, n_features))
                    classes[i] = classes[int(generator.uniform(0,i-1))] + shift * generator.uniform(1,10)
                else:
                    shift = generator.uniform(-5.0, 5.0, size=(1, n_features))
                    classes[i] = classes[i-1] + shift
            # new data portion
            new_data = generator.normal(loc=classes[i], scale=std,
                                        size=(n, n_features))
            new_labels = np.full(n, i)
            # check for lin sep for new class in pair with every another one
            for target_class in range(i):
                target = X[(y == target_class)]
                target_labels = y[(y == target_class)]
                x_ = np.concatenate((target, new_data))
                y_ = np.concatenate((target_labels, new_labels))
                from sklearn.preprocessing import StandardScaler
                sc = StandardScaler()
                x_ = sc.fit_transform(x_)
                from sklearn.linear_model import Perceptron
                perceptron = Perceptron(random_state=0)
                try:
                    perceptron.fit(x_, y_)
                except ValueError:
                    continue
                predicted = perceptron.predict(x_)
                betrayers = 0
                if (linear_separable):
                     for p in range(len(predicted)):
                        if p < len(target_labels):
                            if predicted[p] == i:
                                    target_labels[p] = i
                        else:
                            if predicted[p] == target_class:
                                new_labels[p - len(target_labels)] = target_class
                else:
                    for p in range(len(predicted)):
                        if p < len(target_labels):
                            if predicted[p] == i:
                                betrayers += 1
                        else:
                            if predicted[p] == target_class:
                                betrayers += 1
                    for p in range(len(predicted)):
                        intersection_real = betrayers / len(predicted)
                        if intersection_real <= max_intersection_percentage:
                            break
                        if p < len(target_labels):
                            if predicted[p] == i:
                                target_labels[p] = i
                                betrayers -= 1
                        else:
                            if predicted[p] == target_class:
                                new_labels[p - len(target_labels)] = target_class
                                betrayers -= 1
                y[(y == target_class)] = target_labels
                changed_data   = new_data[(new_labels==target_class)]
                changed_labels = new_labels[(new_labels==target_class)]
                X = np.append(X, changed_data, axis=0)
                y = np.append(y, changed_labels, axis=0)
                new_data = new_data[(new_labels==i)]
                new_labels = new_labels[(new_labels==i)]
            X = np.append(X, new_data, axis=0)
            y = np.append(y, new_labels, axis=0)
    if draw:
        if n_features != 2:
            print("Drawing is available only for 2-features space")
        else:
            plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, s=25, edgecolor='k')
            plt.show()
    return X,y

start = time.time()
X,y = make_dataset(n_features = 2, n_samples = 100000, n_classes = 100,
                   linear_separable=True, n_clusters=4, max_intersection_percentage = 0.01, draw=True)
end = time.time()
print(end - start)

print(len(X))
