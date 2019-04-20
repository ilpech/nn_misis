from project_tools import *

#add sigmoid act func
#add biases

class Perceptron:
    def __init__(self, n_layers=3, n_neurons_per_layer=10, classes=2, features=2, lr=0.1):
        self.__layers = []
        self.n_layers = n_layers
        self.n_neurons_per_layer = n_neurons_per_layer
        self.classes = classes
        self.answer = 'n_classes'
        self.features = features
        self.lr = lr
        self.make_layers()

    def make_layers(self):
        inp_layer = 2 * np.random.random((self.features, self.n_neurons_per_layer)) - 1
        reg_layer = 2 * np.random.random((self.n_neurons_per_layer,
                                          self.n_neurons_per_layer)) - 1
        ans_layer = None
        # TODO: add two anw type 1, k<n
        if self.answer == 'n_classes':
            ans_layer = 2 * np.random.random((self.n_neurons_per_layer, self.classes)) - 1
        if self.answer == 'one_hot_coding':
            ans_layer = 2 * np.random.random((self.n_neurons_per_layer, 1)) - 1
        for layer in range(self.n_layers):
            # todo: add case when only one layer : input -> output
            if layer == 0:
                self.__layers.append(inp_layer)
            else:
                if layer != self.n_layers-1:
                    self.__layers.append(reg_layer)
                else:
                    self.__layers.append(ans_layer)
        return self.__layers

    @staticmethod
    def __nonlin_sigmoid(x, deriv=False):
        if (deriv == True):
            return nonlin_sigmoid(x)*(1 - nonlin_sigmoid(x))
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def __softmax(A):  
        expA = np.exp(A)
        return expA / expA.sum(axis=1, keepdims=True)

    def forward(self, x):
        for layer in self.__layers:
            x = nonlin_sigmoid(np.dot(x, layer))
        return x

    def backward(self, x, y):
        answs = []
        inp = x
        for layer in self.__layers:
            x = self.__nonlin_sigmoid(np.dot(x, layer))
            answs.append(x)

        errors = [0.0] * len(self.__layers)
        deltas = [0.0] * len(self.__layers)

        for layer in reversed(range(len(self.__layers))):
            if layer == len(self.__layers) - 1:
                errors[layer] = y - answs[layer]
                # print(y)
                # print(answs[layer])
                # print(errors[layer])
            else:
                errors[layer] = deltas[layer+1].dot(self.__layers[layer+1].T)
            deltas[layer] = errors[layer] * self.__nonlin_sigmoid(answs[layer], deriv=True)
            # print(deltas[layer])
            # print(self.__layers)
            # sys.exit(0)

        for layer in range(len(self.__layers)):
            if layer != 0:
                self.__layers[layer] += self.lr * answs[layer-1].T.dot(deltas[layer])
            else:
                self.__layers[layer] += self.lr * inp.T.dot(deltas[layer])

        return y - answs[-1]

        # return answs[-1]



p = Perceptron(n_layers=3, n_neurons_per_layer=10, classes=1, features=2, lr=0.1)
X, y = read_data_csv('/repositories/2classes_test.csv')

# print(y)
#
# sys.exit(0)

X = (X - np.mean(X))/np.var(X)

# y = (y - np.mean(y))/np.var(y)
#
# print(y)
#
# sys.exit(0)

# print(len(X))

# X = np.array([[0,0],
#               [0,1],
#               [1,0],
#               [1,1]])
#
# y = np.array([[0,1,1,0]]).T

# X = np.array([  [0,0,1],
#                 [0,1,1],
#                 [1,0,1],
#                 [1,1,1] ])
#
# # выходные данные
# y = np.array([[1,0,1,1]]).T

# print(y)
# sys.exit(0)

# print(np.array([X[0]]))



# p.make_layers()

# np.seterr(all='ignore')

for i in range(100):
    for sample in range(len(X)):
        x_ = np.array([X[sample]])
        y_ = np.array([y[sample]])
        # e = p.backward(x_, y_)

        e = p.backward(X, y)
        print(str(np.mean(np.abs(e))))
# print(np.rint(p.forward(X)) == y)
#
# print(p.forward(np.array([[0,0]])))

        # if (i % 1) == 0:
        #     print("Error:", str(np.mean(np.abs(e))))





X_t, y_t = read_data_csv('/repositories/data.csv')
# X_t, y_t = read_data_csv('/repositories/5classes.csv')
#

X_t_norm = (X_t - np.mean(X_t))/np.var(X_t)
answ = p.forward(X_t_norm)

plt.scatter(X_t[:, 0], X_t[:, 1], marker='o', c=y_t[:,0], s=25, edgecolor='k')
plt.show()

plt.scatter(X_t[:, 0], X_t[:, 1], marker='o', c=np.rint(answ)[:,0], s=25, edgecolor='k')
plt.show()

# print(y_t)
# print(answ)

print(y_t==np.rint(answ))

sys.exit(0)

# draw_dataset(X,y)
# sys.exit(0)

# случайно инициализируем веса, в среднем - 0
w0 = 2 * np.random.random((2, 2)) - 1
w1 = 2 * np.random.random((2, 2)) - 1
w2 = 2 * np.random.random((2, 1)) - 1

np.seterr(all='ignore')

for j in range(100000):
    l0 = X
    l1 = nonlin_sigmoid(np.dot(l0, w0))
    l2 = nonlin_sigmoid(np.dot(l1, w1))
    l3 = nonlin_sigmoid(np.dot(l2, w2))

    l3_error = y - l3

    if (j % 10000) == 0:
        print("Error:", str(np.mean(np.abs(l3_error))))

    l3_delta = l3_error * nonlin_sigmoid(l3, deriv=True)

    l2_error = l3_delta.dot(w2.T)
    l2_delta = l2_error * nonlin_sigmoid(l2, deriv=True)

    l1_error = l2_delta.dot(w1.T)
    l1_delta = l1_error * nonlin_sigmoid(l1, deriv=True)



    w2 += l2.T.dot(l3_delta)
    w1 += l1.T.dot(l2_delta)
    w0 += l0.T.dot(l1_delta)

#
# sys.exit(0)

# l0 = X_t
# l1 = nonlin_sigmoid(np.dot(l0, w0))
# l2 = nonlin_sigmoid(np.dot(l1, w1))
# print(np.rint(l2))
    # print(w0,w1)
    # sys.exit(0)
