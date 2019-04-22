#brief: Dataset generator for nn testing
#author: pichugin
#usage: python3 dataset_generator.py --dst /repositories --name dataset_sample --n-features 2 --n-samples 100000 --n-classes 40 --linear-separable True --n-clusters 1 --max-intersection-percentage 0.01 --draw True --save True

import numpy as np
from project_tools import *

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
                    shift = generator.uniform(-7.0, 7.0, size=(1, n_features))
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
                    if euclidean_distance(classes[i], classes[target_class]) > 15.0:
                        continue
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

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string, use False or True')
    return s == 'True'

def dataset_to_csv(X,y,dst_dir,file_name):
    dst = os.path.join(dst_dir, file_name + '.csv')
    Xy = np.column_stack((X,y))
    np.savetxt(dst, Xy, fmt='%f',delimiter=',')
    print('Datasets was saved: ', dst)


parser = argparse.ArgumentParser(
                                description=('Dataset generator for nn testing')
                                )
parser.add_argument('--dst', type=str, default='')
parser.add_argument('--name', type=str, default='')
parser.add_argument('--n-features', type=int, default=2)
parser.add_argument('--n-samples', type=int, default=10000)
parser.add_argument('--n-classes', type=int, default=10)
parser.add_argument('--linear-separable', type=boolean_string, default=True)
parser.add_argument('--n-clusters', type=int, default=5)
parser.add_argument('--max-intersection-percentage', type=float, default=0.1)
parser.add_argument('--draw', type=boolean_string, default=True)
parser.add_argument('--save', type=boolean_string, default=False)

opt = parser.parse_args()
if opt.save:
    if not os.path.isdir(opt.dst):
        print('Check dst {} path for saving data'.format(opt.dst))
        sys.exit(0)

start = time.time()
X,y = make_dataset(n_features=opt.n_features, n_samples=opt.n_samples, n_classes=opt.n_classes,
                   linear_separable=opt.linear_separable, n_clusters=opt.n_clusters,
                   max_intersection_percentage = opt.max_intersection_percentage, draw=opt.draw)
end = time.time()

print("Generation time {:03f}".format(end - start))

if opt.save:
    dataset_to_csv(X,y, opt.dst, opt.name)
