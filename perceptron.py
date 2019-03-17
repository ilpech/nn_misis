from project_tools import *
import numpy as np


def read_data_csv(path):
    try:
        if not os.path.isfile(path):
            raise NameError
    except NameError:
        print('Check csv path', path)
    data = np.genfromtxt(path, delimiter=',')
    X = data[:,:-1]
    y = data[:,-1:]
    return X, y



X, y = read_data_csv('/repositories/data_2_classes.csv')

X_t, y_t = read_data_csv('/repositories/data.csv')



# draw_dataset(X,y)
# sys.exit(0)

# случайно инициализируем веса, в среднем - 0
w0 = 2 * np.random.random((2, 5)) - 1
w1 = 2 * np.random.random((5, 1)) - 1

# print(w0, w1)
#

for j in range(100000):
    l0 = X
    l1 = nonlin_sigmoid(np.dot(l0, w0))
    l2 = nonlin_sigmoid(np.dot(l1, w1))

    l2_error = y - l2

    if (j % 10000) == 0:
        print("Error:", str(np.mean(np.abs(l2_error))))

    l2_delta = l2_error * nonlin_sigmoid(l2, deriv=True)
    l1_error = l2_delta.dot(w1.T)

    l1_delta = l1_error * nonlin_sigmoid(l1, deriv=True)

    w1 += l1.T.dot(l2_delta)
    w0 += l0.T.dot(l1_delta)



l0 = X_t
l1 = nonlin_sigmoid(np.dot(l0, w0))
l2 = nonlin_sigmoid(np.dot(l1, w1))
print(np.rint(l2))
    # print(w0,w1)
    # sys.exit(0)
