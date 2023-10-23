import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import scipy as sc

# Importing the data
dataset = sc.io.loadmat('Home-exam\ExamData3D.mat')

# Splitting the data into training and test sets
X_train = dataset['X_train']
X_test = dataset['X_test']
X_train, X_test = np.delete(X_train, 2, 0), np.delete(X_test, 2, 0)
Y_train = dataset['Y_train'][0]
Y_test = dataset['Y_test'][0]

# adding the extra 'dimension' to X_train
X_train = np.insert(X_train, 0, 1, axis=0)
X_test = np.insert(X_test, 0, 1, axis=0)

# defining parameters
#rho = 0.0001 # learning rate

# new Y_train arrays
Y_train_class_1 = np.zeros(len(Y_train))
Y_train_class_2 = np.zeros(len(Y_train))
Y_train_class_3 = np.zeros(len(Y_train))

Y_train_class_1[Y_train == 1] = 1
Y_train_class_2[Y_train == 2] = 1
Y_train_class_3[Y_train == 3] = 1

class perceptron:
    def __init__(self, c, y, N=len(X_train[0])):
        self.w = np.random.rand(3,1) # random initial weights
        self.rho = rho
        self.y_hat = np.zeros(N)
        self.c = c
        self.y = np.zeros(N)
        self.y[y == self.c] = 1

    def forward(self, x):
        g = self.w.T @ x
        f = 1/(1+np.exp(-g))
        self.y_hat = f
    
    def backward(self, x):
        df = 1/(1+np.exp(-self.w.T @ x))*(1-(1/(1+np.exp(-self.w.T @ x))))
        dw = (self.rho*(self.y-self.y_hat)*df @ x.T).T
        self.w += dw

class neural_network:
    def __init__(self, X_train, Y_train, X_test, Y_test):
        self.accuracy = 0
        self.predictions = np.zeros(len(Y_test))
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.neurons = [perceptron(1, self.Y_train), perceptron(2, self.Y_train), perceptron(3, self.Y_train)]

    def train(self, T):
        for t in range(T):
            for neuron in self.neurons:
                neuron.forward(self.X_train)
                neuron.backward(self.X_train)
        
    def predict(self):
        for neuron in self.neurons:
            neuron.forward(self.X_test)
        self.predictions = np.argmax([self.neurons[0].y_hat, self.neurons[1].y_hat, self.neurons[2].y_hat], axis=0)+1
        self.accuracy = np.sum(self.predictions == self.Y_test)/len(self.Y_test)
        return self.accuracy
    
    def decision_boundary(self, resolution=100):
        x, y = np.linspace(-4.0, 3.7, num=resolution), np.linspace(-3.0, 4.2, num=resolution)
        XY = np.asarray(np.meshgrid(x,y)).reshape(2, -1)
        XY = np.insert(XY, 0, 1, axis=0)
        for neuron in self.neurons:
            neuron.forward(XY)
        predictions = np.argmax([self.neurons[0].y_hat, self.neurons[1].y_hat, self.neurons[2].y_hat], axis=0)+1
        plt.pcolormesh(x, y, predictions.reshape(resolution, resolution), cmap = 'Set1', alpha=0.6, vmin=0, vmax=12)
        plt.scatter(self.X_test[1, self.Y_train == 1], self.X_test[2, self.Y_train == 1], color='red', label='Class 1')
        plt.scatter(self.X_test[1, self.Y_train == 2], self.X_test[2, self.Y_train == 2], color='blue', label='Class 2')
        plt.scatter(self.X_test[1, self.Y_train == 3], self.X_test[2, self.Y_train == 3], color='green', label='Class 3')
        plt.title("Decision boundary")
        plt.show()

rho = 0.001
nn = neural_network(X_train, Y_train, X_test, Y_test)
T = np.linspace(1, 16, 16, dtype=int)
print("rho = ", rho)
for t in T:
    nn.train(2**t)
    print("Accuracy for T = ", 2**t, " is ", nn.predict())
nn.decision_boundary()
