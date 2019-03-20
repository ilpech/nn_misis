from project_tools import *

class Perceptron:

    def __init__(self, n_layers=3, n_neurons_per_layer=100, classes=2):
        self.__layers = []
        self.n_layers = n_layers
        self.n_neurons_per_layer = n_neurons_per_layer
        self.classes = classes
        self.answer = 'n_classes'
        # super(Perceptron, self).__init__(n_layers, n_neurons_per_layer, classes)
        self.make_layers()

    def make_layers(self):
        inp_layer = 2 * np.random.random((self.classes, self.n_neurons_per_layer)) - 1
        reg_layer = 2 * np.random.random((self.n_neurons_per_layer, self.n_neurons_per_layer)) - 1
        ans_layer = None
        # TODO: add two anw type 1, k<n
        if self.answer == 'n_classes':
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
        try:
            return 1 / (1 + np.exp(-x))
        except RuntimeWarning:
            print(x)

    @staticmethod
    def nonlin_relu(x, deriv=False):
        print(x)
        for i in range(len(x)):
            if (deriv == True):
                if x[i] > 0:
                    x[i] = 1
                else:
                    x[i] = 0
            if x[i] > 0:
                x[i] = x[i]
            else:
                x[i] =0
        return x

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
            else:
                errors[layer] = deltas[layer+1].dot(self.__layers[layer+1].T)
            deltas[layer] = errors[layer] * self.__nonlin_sigmoid(answs[layer], deriv=True)

        for layer in range(len(self.__layers)):
            if layer != 0:
                self.__layers[layer] += answs[layer-1].T.dot(deltas[layer])
            else:
                self.__layers[layer] += inp.T.dot(deltas[layer])

        return y - answs[-1]



p = Perceptron(n_layers=3, n_neurons_per_layer=2, classes=2)
X, y = read_data_csv('/repositories/data_2_classes.csv')

print(X.shape)

# p.make_layers()

np.seterr(all='ignore')

for i in range(10000000):
    e = p.backward(X, y)
    if (i % 10000) == 0:
        print("Error:", str(np.mean(np.abs(e))))

sys.exit(0)


X, y = read_data_csv('/repositories/data_2_classes.csv')
X_t, y_t = read_data_csv('/repositories/data.csv')

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
