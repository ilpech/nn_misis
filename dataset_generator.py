import matplotlib.pyplot as plt
import numpy as np

center_box=(0, 100.0)
cluster_std=2.0
n_features = 2
n_samples = 200
n_classes = 4

generator = np.random.RandomState(420)

classes = generator.uniform(center_box[0], center_box[1], size=(n_classes, n_features))

# X = [] # values
# y = [] # labels
n_samples_per_class = [int(n_samples // n_classes)] * n_classes

for i in range(n_samples % n_classes):
    n_samples_per_class[i] += 1

cluster_std = np.full(len(classes), cluster_std)

X = []
y = []
for i, (n, std) in enumerate(zip(n_samples_per_class, cluster_std)):
    if(i == 0):
        X.append(generator.normal(loc=classes[i], scale=std,
                                  size=(n, n_features)))
        y += [i] * n
    else:
        added_X = np.concatenate(X)
        added_Y = np.array(y)
        linear_sep = False
        # check for linear sep
        for target_class in range(i):
            target = added_X[(added_Y == target_class)]
            target_labels = added_Y[(added_Y == target_class)]
            new_data = generator.normal(loc=classes[i], scale=std,
                                  size=(n, n_features))
            new_labels = np.full(n, i)
            x_ = np.concatenate((target, new_data))
            y_ = np.concatenate((target_labels, new_labels))

            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            x_ = sc.fit_transform(x_)
            from sklearn.linear_model import Perceptron
            perceptron = Perceptron(random_state=0)
            perceptron.fit(x_, y_)
            predicted = perceptron.predict(x_)

            print(predicted)

            linear_sep = True

            # sys.exit(0)
        if linear_sep:
            X.append(generator.normal(loc=classes[i], scale=std,
                                      size=(n, n_features)))
            y += [i] * n



X = np.concatenate(X)
y = np.array(y)


# print(X.shape)
#
# print(X , y.shape)

plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, s=25, edgecolor='k')
plt.show()