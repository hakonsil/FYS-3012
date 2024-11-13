import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import scipy as sc
from sklearn import metrics

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

class perceptron:
    def __init__(self, dimension, N=len(X_train[0])):
        """
        Perceptron
        --------
        Parameters
        ----
        - dimension: dimension of the feature vectors
        - N: number of training feature vectors
        """
        # initializing random weights, and all other attributes to zero
        self.w = np.random.rand(dimension,1)
        self.dw = np.zeros(self.w.shape)
        self.y_hat = np.zeros(N)
        self.v = np.zeros(N)
        self.e = np.zeros(N)
        self.delta = 0

    def forward(self, x):
        """
        Forward pass (calculates the output of the perceptron)

        Parameters
        ----
        - x: array of feature vectors
        """
        self.v = self.w.T @ x
        self.y_hat = 1/(1+np.exp(-self.v))

    def calculate_e(self, deltas, weights):
        """
        Calculate the error of the perceptron

        Parameters
        ----------
        - deltas: the deltas of the next layer
        - weights: the weights of the next layer (that are connected to the perceptron)
        """
        self.e = (deltas @ weights.T).T

    def calculate_delta(self):
        """
        Calculates the value for delta
        """
        f = 1/(1+np.exp(-self.v))
        df = f*(1-f)
        self.delta = np.multiply(self.e,df)

    def backward(self, y):
        """
        Backward pass (updates the weights of the perceptron)

        Parameters
        ----
        - y: array of outputs from the previous layer
        """
        self.dw = alpha*self.dw - (rho*y@self.delta.T) # (momentum term + cost function term)
        self.w += self.dw

class output_perceptron(perceptron):
    def __init__(self, label, dimension, y, N=len(X_train[0])):
        """
        Output perceptron (only difference from regular perceptron if calculating e and delta)
        
        Parameters
        ----
        - label: the label of the output perceptron
        - dimension: dimension of the feature vectors
        - y: training set
        - N: number of training feature vectors
        """
        super().__init__(dimension, N=len(X_train[0]))
        self.y = np.zeros(N)
        self.y[y == label] = 1

    def calculate_e(self):
        """
        Calculates the error
        """
        self.e = self.y_hat - self.y

    def calculate_delta(self):
        """
        Calculates the value for delta
        """
        f = 1/(1+np.exp(-self.v))
        df = f*(1-f)
        self.delta = self.e*df

class MLP:
    def __init__(self, shape, X_train, Y_train):
        """
        Multi-layer perceptron
        --------
        Parameters
        ----
        - shape: the shape of the hidden layers of the network, input layer is 2 since the dataset is 2d and output layer is 3 nodes from the task description
        - X_train: training set
        - Y_train: training labels
        """

        # initializing attributes
        self.shape = shape
        self.X_train = X_train
        self.Y_train = Y_train
        self.layers = []
        layer = []

        # creating the layers
        # first hidden layer
        for i in range(shape[0]):
            layer.append(perceptron(3))
        self.layers.append(layer)

        # other hidden layers
        for i in range(1, len(shape)):
            layer = []
            for j in range(shape[i]):
                layer.append(perceptron(shape[i-1]+1))
            self.layers.append(layer)

        # output layer
        output_layer = [output_perceptron(1, shape[-1]+1, self.Y_train), output_perceptron(2, shape[-1]+1, self.Y_train), output_perceptron(3, shape[-1]+1, self.Y_train)]
        self.layers.append(output_layer)

    def forward_propagation(self, X_train):
        """
        Propogates the network forwards (calculates the output of the network)

        Parameters
        ----------
        - X_train: training set (input to the network)
        """

        # inpput the training set into the first layer
        for neuron in self.layers[0]:
            neuron.forward(X_train)

        # propogate the output of the first layer to the next layer, and so on
        for i in range(1, len(self.layers)):
            input = []

            for n in self.layers[i-1]: # for all neurons in the previous layer
                input.append(n.y_hat.tolist()[0]) # add the output to the input list
            input = np.insert(input, 0, 1, axis=0) # add the bias term

            for neuron in self.layers[i]: # propogate the input to the next layer
                neuron.forward(np.asarray(input))
    
    def backward_propagation(self):
        """
        Propogates the network backwards (updates the weights of the network)
        """

        # calculate e and delta for the last layer
        for neuron in self.layers[-1]:
            neuron.calculate_e()
            neuron.calculate_delta()

        for i in reversed(range(0, len(self.layers)-1)): # iterate backwards through the layers
            for j in range(len(self.layers[i])): # iterate through the neurons in the layer
                deltas = []
                weights = []
                for neuron in self.layers[i+1]: # for all neurons in the next layer

                    deltas.append(neuron.delta) # add the delta to the list
                    weights.append(neuron.w[j+1]) # add the weights to the list
                deltas, weights = np.matrix(np.asarray(deltas)).T, np.matrix(np.asarray(weights)).T # convert to matrices

                self.layers[i][j].calculate_e(deltas, weights) # calculate the new error for the neuron
                self.layers[i][j].calculate_delta() # calculate the new delta for the neuron

        # update the weights for the first layer
        for neuron in self.layers[0]:
            neuron.backward(self.X_train)

        # update the weights for the other layers
        for i in range(1, len(self.layers)):
            for j in range(len(self.layers[i])):
                y = []
                for neuron in self.layers[i-1]:
                    y.append(neuron.y_hat)
                y = np.asarray(y)
                y = np.insert(y, 0, 1, axis=0) # bias term
                self.layers[i][j].backward(np.matrix(y))
    
    def train(self, T, X_train, Y_train,X_test, Y_test, predict=False):
        """
        Training the network (updates the weights T times)

        Parameters
        ----
        - T: number of epochs
        - X_train: training set
        - Y_train: training labels
        - X_test: test set
        - Y_test: test labels
        - predict: (bool) if we want to get the accuracy for each epoch
        """

        # if we want to get the accuracy for each epoch
        if predict == True:
            predictions_train = np.zeros(T)
            predictions_test = np.zeros(T)
            for t in range(T):
                # Just for checking the progress
                if t % 500 == 0:
                    print(t)
                self.forward_propagation(self.X_train)
                self.backward_propagation()
                self.predict(X_train, Y_train)
                predictions_train[t] = self.accuracy
                self.predict(X_test, Y_test)
                predictions_test[t] = self.accuracy
            return predictions_train, predictions_test
        
        # if we only want the final accuracy
        else:
            for t in range(T):
                self.forward_propagation(self.X_train)
                self.backward_propagation()
    
    def predict(self, X_test, Y_test):
        """
        Predicting the labels of the test set

        Parameters
        ----
        - X_test: test set
        - Y_test: test labels
        """
        # calculate the output of the network
        self.forward_propagation(X_test)

        # taking the most probable class
        self.prediction = np.argmax([self.layers[-1][0].y_hat, self.layers[-1][1].y_hat, self.layers[-1][2].y_hat], axis=0)+1

        # calculating the accuracy
        self.accuracy = np.sum(self.prediction == Y_test)/len(Y_test)

    def decision_boundary(self, Y_test,resolution=100):
        """
        Plotting the decision boundary
        
        Parameters
        ----------
        - Y_test: The true labels of the test set
        - resolution: The resolution of the decision boundary
        """

        # creating the meshgrid
        x, y = np.linspace(-4.0, 4.2, num=resolution), np.linspace(-4.0, 4.2, num=resolution)
        XY = np.asarray(np.meshgrid(x,y)).reshape(2, -1)
        XY = np.insert(XY, 0, 1, axis=0)

        # calculating the output of the network
        self.forward_propagation(XY)

        # predicting the labels
        predictions = np.argmax([self.layers[-1][0].y_hat, self.layers[-1][1].y_hat, self.layers[-1][2].y_hat], axis=0)+1

        # plotting the decision boundary
        plt.pcolormesh(x, y, predictions.reshape(resolution, resolution), cmap = 'Set1', alpha=0.6, vmin=0, vmax=12)
        plt.scatter(X_test[1, Y_test == 1], X_test[2, Y_test == 1], marker = '+', color='red', label='Class 1')
        plt.scatter(X_test[1, Y_test == 2], X_test[2, Y_test == 2], marker = '+', color='blue', label='Class 2')
        plt.scatter(X_test[1, Y_test == 3], X_test[2, Y_test == 3], marker = '+', color='green', label='Class 3')
        plt.xlabel("$x_1$")
        plt.ylabel("$x_2$")

    def plot_confusion_matrix(self, Y_test, title):
        """
        Plotting the confusion matrix for the prediction
        
        Parameters
        ----------
        - Y_test: The true labels of the test set
        - title: The title of the plot
        """

        confusion_matrix = metrics.confusion_matrix(Y_test, self.prediction[0])
        confusion_matrix_display = metrics.ConfusionMatrixDisplay(confusion_matrix)
        confusion_matrix_display.plot(cmap="Reds")
        plt.title("Confusion matrix for " + title)
        confusion_matrix_display.ax_.set_xticks([])
        confusion_matrix_display.ax_.set_yticks([])
        plt.show()

if __name__ == "__main__":
    # defining the parameters
    rho = 0.008
    alpha = 0.65
    shape = [10, 8, 6]
    T = 2500

    # creating and training the network
    nn = MLP(shape, X_train, Y_train)
    accuracies_train, accuracies_test = nn.train(T, X_train, Y_train, X_test, Y_test, predict=True)

    # plotting the decision boundary
    print("Test set: ", np.round(accuracies_test[-1], 3), ", Training set: ", np.round(accuracies_train[-1], 3))
    plt.title(chr(961) + "=" + str(rho) + " " + chr(945) + "=" + str(alpha))
    nn.decision_boundary(Y_test, resolution = 200)
    plt.show()

    # plotting the accuracy as a function of T
    plt.plot(np.arange(T), accuracies_train, label="Training set", color = 'blue')
    plt.plot(np.arange(T), accuracies_test, label="Test set", color = 'red')
    plt.xscale("log")
    plt.title("Accuracy as a function of T")
    plt.xlabel("T")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    # plotting the confusion matrix
    nn.plot_confusion_matrix(Y_test, " Test set")