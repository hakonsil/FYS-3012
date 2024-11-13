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

# adding the extra 'dimension' to X_train and X_test
X_train = np.insert(X_train, 0, 1, axis=0)
X_test = np.insert(X_test, 0, 1, axis=0)

class perceptron:
    def __init__(self, c, y, N=len(X_train[0])):
        """
        Perceptron
        --------

        Parameters
        ----
        - c: class which the perceptron is trained to recognize
        - y: training set
        - N: number of training feature vectors
        """

        self.w = np.random.rand(3,1) # random initial weights
        self.y_hat = np.zeros(N)
        self.c = c 
        self.y = np.zeros(N)
        self.y[y == self.c] = 1 # y = 1 if the feature vector belongs to class c, 0 else

    def forward(self, x):
        """
        Forward pass (calculates the output of the perceptron)

        Parameters
        ----
        - x: array of feature vectors
        """

        g = self.w.T @ x # discriminant function
        f = 1/(1+np.exp(-g)) # activation function
        self.y_hat = f

    def backward(self, x):
        """
        Backward pass (updates the weights of the perceptron)

        Parameters
        ----
        - x: array of feature vectors
        """

        df = 1/(1+np.exp(-self.w.T @ x))*(1-(1/(1+np.exp(-self.w.T @ x)))) # derivative of the activation function
        dw = (rho*(self.y-self.y_hat)*df @ x.T).T # weight update
        self.w += dw

class neural_network:
    def __init__(self, X_train, Y_train, X_test, Y_test):
        """
        Neural network (single layer with 3 nodes)

        Parameters
        ----
        - X_train: training set
        - Y_train: training labels
        - X_test: test set
        - Y_test: test labels
        """

        self.accuracy = 0
        self.predictions = np.zeros(len(Y_test))
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test

        # setting up the perceptrons
        self.neurons = [perceptron(1, self.Y_train), perceptron(2, self.Y_train), perceptron(3, self.Y_train)]

    def train(self, T):
        """
        Training the neural network (updates the weights T times)

        Parameters
        ----
        - T: number of epochs
        """

        for t in range(T):
            for neuron in self.neurons:
                neuron.forward(self.X_train)
                neuron.backward(self.X_train)

    def predict(self):
        """
        Predicting the labels of the test set

        Returns
        ----
        - accuracy: accuracy of the predictions
        """

        for neuron in self.neurons:
            neuron.forward(self.X_test) # forward pass to get the predictions
        
        # taking the most probable class
        self.predictions = np.argmax([self.neurons[0].y_hat, self.neurons[1].y_hat, self.neurons[2].y_hat], axis=0)+1
        self.accuracy = np.sum(self.predictions == self.Y_test)/len(self.Y_test)

        return self.accuracy

    def decision_boundary(self, txt, resolution=120):
        """
        Plotting the decision boundary

        Parameters
        ----
        - txt: title of the plot
        - resolution: resolution of the decision boundary
        """

        x, y = np.linspace(-4.0, 4.2, num=resolution), np.linspace(-4.0, 4.2, num=resolution)
        XY = np.asarray(np.meshgrid(x,y)).reshape(2, -1)
        XY = np.insert(XY, 0, 1, axis=0)
        for neuron in self.neurons:
            neuron.forward(XY)
        predictions = np.argmax([self.neurons[0].y_hat, self.neurons[1].y_hat, self.neurons[2].y_hat], axis=0)+1
        plt.pcolormesh(x, y, predictions.reshape(resolution, resolution), cmap = 'Set1', alpha=0.6, vmin=0, vmax=12)
        plt.scatter(self.X_test[1, self.Y_train == 1], self.X_test[2, self.Y_train == 1], marker = '+', color='red', label='Class 1')
        plt.scatter(self.X_test[1, self.Y_train == 2], self.X_test[2, self.Y_train == 2], marker = '+', color='blue', label='Class 2')
        plt.scatter(self.X_test[1, self.Y_train == 3], self.X_test[2, self.Y_train == 3], marker = '+', color='green', label='Class 3')
        plt.title("Single layer Perceptron: " + txt)
        plt.xlabel("$x_1$")
        plt.ylabel("$x_2$")

    def plot_confusion_matrix(self, title):
        """
        Plotting the confusion matrix

        Parameters
        ----
        - title: title of the plot
        """

        confusion_matrix = metrics.confusion_matrix(self.Y_test, self.predictions[0])
        confusion_matrix_display = metrics.ConfusionMatrixDisplay(confusion_matrix)
        confusion_matrix_display.plot(cmap="Reds")
        plt.title("Single layer Perceptron: " + title)
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        confusion_matrix_display.ax_.set_xticks([])
        confusion_matrix_display.ax_.set_yticks([])
        plt.show()

if __name__ == "__main__":
    """---Testing the neural network---"""
    # defining parameters
    rho = 0.01 # learning rate
    T = np.linspace(0, 200, 201, dtype=int) # epochs

    # initiating the neural network, for both training and test sets
    NN_test = neural_network(X_train, Y_train, X_test, Y_test)
    NN_train = neural_network(X_train, Y_train, X_train, Y_train)

    # get the accuracy as a function of T
    test_accuracies = np.zeros(len(T))
    train_accuracies = np.zeros(len(T))
    for t in range(len(T)):
        #print(t)
        NN_test.train(T[t])
        NN_train.train(T[t])
        test_accuracies[t] = NN_test.predict()
        train_accuracies[t] = NN_train.predict()

    # plotting the decision boundary
    plt.subplot(1,2,1)
    NN_test.decision_boundary("Test set" + " (" + str(chr(961)) + "=" + str(rho)+ ")" + ", T = " + str(T[-1]))
    plt.subplot(1,2,2)
    NN_train.decision_boundary("Training set")
    plt.show()
    print("Training set: ", np.round(train_accuracies[-1], 3), ", Test set: ", np.round(test_accuracies[-1], 3))

    # plotting the accuracy as a function of T
    plt.plot(T, test_accuracies, label="Test set")
    plt.plot(T, train_accuracies, label="Training set")
    plt.title("Accuracy as a function of T (" + str(chr(961)) + "=" + str(rho)+ ")")
    plt.xlabel("T")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
    print("Highest accuracy for test set: ", np.round(np.max(test_accuracies), 3), ", at T = ", np.argmax(test_accuracies)+1)

    # plotting the confusion matrix
    NN_test.plot_confusion_matrix("Test set")
    NN_train.plot_confusion_matrix("Training set")