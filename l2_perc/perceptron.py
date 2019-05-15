#usage: python3 perceptron.py

from project_tools import *

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
        if self.answer == 'n_classes':
            ans_layer = 2 * np.random.random((self.n_neurons_per_layer, self.classes)) - 1
        if self.answer == 'one_hot_coding':
            ans_layer = 2 * np.random.random((self.n_neurons_per_layer, 1)) - 1
        for layer in range(self.n_layers):
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

            else:
                errors[layer] = deltas[layer+1].dot(self.__layers[layer+1].T)
            deltas[layer] = errors[layer] * self.__nonlin_sigmoid(answs[layer], deriv=True)

        for layer in range(len(self.__layers)):
            if layer != 0:
                self.__layers[layer] += self.lr * answs[layer-1].T.dot(deltas[layer])
            else:
                self.__layers[layer] += self.lr * inp.T.dot(deltas[layer])

        return y - answs[-1]



p = Perceptron(n_layers=3, n_neurons_per_layer=10, classes=1, features=2, lr=0.1)
print(os.getcwd())
X, y = read_data_csv('data/2_sep.mat')


X = (X - np.mean(X))/np.var(X)


scores = []

for i in range(1000):
    for sample in range(len(X)):
        x_ = np.array([X[sample]])
        y_ = np.array([y[sample]])
        # e = p.backward(x_, y_)
        e = p.backward(X, y)
        
        scores.append(str(np.mean(np.abs(e))))

e = 0
for i in range(len(scores)):
    if i == 1:
        print("Error: ", scores[i])
    if i % 100000:
        e += 1
        print("Epoch: {} Error: {}".format(int(e), scores[i]))






X_t, y_t = read_data_csv('data/2_sep_data.mat')

X_t_norm = (X_t - np.mean(X_t))/np.var(X_t)
answ = p.forward(X_t_norm)

plt.scatter(X_t[:, 0], X_t[:, 1], marker='o', c=y_t[:,0], s=25, edgecolor='k')
plt.show()

plt.scatter(X_t[:, 0], X_t[:, 1], marker='o', c=np.rint(answ)[:,0], s=25, edgecolor='k')
plt.show()


print(y_t==np.rint(answ))

sys.exit(0)
