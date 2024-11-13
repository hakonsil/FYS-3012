import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import scipy as sc
from sklearn import metrics
from prettytable import PrettyTable

# Importing the data
dataset = sc.io.loadmat('Home-exam\ExamData3D.mat')

# Splitting the data into training and test sets
X_train = dataset['X_train']
X_test = dataset['X_test']
Y_train = dataset['Y_train'][0]
Y_test = dataset['Y_test'][0]

# defining the probability for the classes
P = 1/3 # using the a priori knowledge that the classes are equiprobable (to do properly it should be calculated from the data set)

def mean_vector(x, y, class_nr):
    """
    Calculates the mean vector for a given class
    
    Parameters
    ----------
    - x : The set of feature vectors
    - y: Known labels for the feature vectors
    - class_nr: The class for which the mean vector is calculated

    Returns
    -------
    - mu: The mean vector for the given class
    """

    # special case for 1 dimension (only difference is the indexing of the array)
    if x.ndim == 1:
        mu = np.mean(x[y == class_nr]) # calculating the mean vector for the given class
    else:
        mu = np.mean(x[:,y == class_nr], axis = 1) # calculating the mean vector for the given class

    return np.matrix(mu).T # returning the mean vector as a column matrix

def J3_score(x, y):
    """
    Calculates the modified J3 score for a given data set

    Parameters
    ----------
    - x: The set of feature vectors
    - y: Known labels for the feature vectors

    Returns
    -------
    - J3: The modified J3 score for the given data set
    """

    # calculating the mean vector for each class
    mu1 = mean_vector(x, y, 1)
    mu2 = mean_vector(x, y, 2)
    mu3 = mean_vector(x, y, 3)

    # calculating the global mean vector
    mu0 = np.matrix((mu1 + mu2 + mu3)*P)

    # calculating the covariance matrix for each class
    if x.ndim == 1:
        cov1 = np.cov(np.array((x[y == 1])))
        cov3 = np.cov(np.array((x[y == 3])))
        cov2 = np.cov(np.array((x[y == 2])))
    else:
        cov1 = np.cov(np.array((x[:,y == 1])))
        cov3 = np.cov(np.array((x[:,y == 3])))
        cov2 = np.cov(np.array((x[:,y == 2])))

    # calculating the within class scatter matrix
    Sw = (cov1 + cov2 + cov3)*P

    # calculating the between class scatter matrix
    Sb  = ((mu1-mu0)@((mu1-mu0).T) + (mu2-mu0)@((mu2-mu0).T) + (mu3-mu0)@((mu3-mu0).T))*P

    # calculating the modified J3 score
    if x.ndim == 1:
        J3 = Sb/Sw
    else:
        J3 = np.trace(np.linalg.inv(Sw)@Sb)
    return J3

def print_J3():
    """
    Prints the J3 score for each dimension and combination of dimensions
    """
    print("------------Task 1------------")
    table = PrettyTable()
    table.field_names = ["Dimension", "J3 score"]
    table.add_row(["[1]", np.round(float(J3_score(X_train[0], Y_train)), 3)])
    table.add_row(["[2]", np.round(float(J3_score(X_train[1], Y_train)), 3)])
    table.add_row(["[3]", np.round(float(J3_score(X_train[2], Y_train)), 3)])
    table.add_row(["[1,2]", np.round(float(J3_score(np.delete(X_train, 2, 0), Y_train)), 3)])
    table.add_row(["[1,3]", np.round(float(J3_score(np.delete(X_train, 1, 0), Y_train)), 3)])
    table.add_row(["[2,3]", np.round(float(J3_score(np.delete(X_train, 0, 0), Y_train)), 3)])
    table.add_row(["[1,2,3]", np.round(float(J3_score(X_train, Y_train)), 5)])
    print(table)

def plot_data():
    """
    Plots all combinations of dimensions of the training set
    """
    plt.subplot(1,3,1)
    plt.hist(X_train[0][Y_train == 1], bins = 20, alpha = 0.5, label = 'Class 1', color = 'red')
    plt.hist(X_train[0][Y_train == 2], bins = 20, alpha = 0.5, label = 'Class 2', color = 'blue')
    plt.hist(X_train[0][Y_train == 3], bins = 20, alpha = 0.5, label = 'Class 3', color = 'green')
    plt.legend()
    plt.title('Dimension: [1], $J_3$ = ' + str(np.round(float(J3_score(X_train[0], Y_train)), 5)))

    plt.subplot(1,3,2)
    plt.hist(X_train[1][Y_train == 1], bins = 20, alpha = 0.5, label = 'Class 1', color = 'red')
    plt.hist(X_train[1][Y_train == 2], bins = 20, alpha = 0.5, label = 'Class 2', color = 'blue')
    plt.hist(X_train[1][Y_train == 3], bins = 20, alpha = 0.5, label = 'Class 3', color = 'green')
    plt.legend()
    plt.title('Dimension: [2], $J_3$ = ' + str(np.round(float(J3_score(X_train[1], Y_train)), 5)))

    plt.subplot(1,3,3)
    plt.hist(X_train[2][Y_train == 1], bins = 20, alpha = 0.5, label = 'Class 1', color = 'red')
    plt.hist(X_train[2][Y_train == 2], bins = 20, alpha = 0.5, label = 'Class 2', color = 'blue')
    plt.hist(X_train[2][Y_train == 3], bins = 20, alpha = 0.5, label = 'Class 3', color = 'green')
    plt.legend()
    plt.title('Dimension: [3], $J_3$ = ' + str(np.round(float(J3_score(X_train[2], Y_train)), 5)))
    plt.show()

    # plotting the 2D data
    plt.subplot(1,3,1)
    plt.scatter(X_train[0][Y_train == 1], X_train[1][Y_train == 1], label = 'Class 1', color = 'red')
    plt.scatter(X_train[0][Y_train == 2], X_train[1][Y_train == 2], label = 'Class 2', color = 'blue')
    plt.scatter(X_train[0][Y_train == 3], X_train[1][Y_train == 3], label = 'Class 3', color = 'green')
    plt.legend()
    plt.title('Dimensions: [1,2], $J_3$ = ' + str(np.round(float(J3_score(np.delete(X_train, 2, 0), Y_train)), 5)))

    plt.subplot(1,3,2)
    plt.scatter(X_train[0][Y_train == 1], X_train[2][Y_train == 1], label = 'Class 1', color = 'red')
    plt.scatter(X_train[0][Y_train == 2], X_train[2][Y_train == 2], label = 'Class 2', color = 'blue')
    plt.scatter(X_train[0][Y_train == 3], X_train[2][Y_train == 3], label = 'Class 3', color = 'green')
    plt.legend()
    plt.title('Dimensions: [1,3], $J_3$ = ' + str(np.round(float(J3_score(np.delete(X_train, 1, 0), Y_train)), 5)))

    plt.subplot(1,3,3)
    plt.scatter(X_train[1][Y_train == 1], X_train[2][Y_train == 1], label = 'Class 1', color = 'red')
    plt.scatter(X_train[1][Y_train == 2], X_train[2][Y_train == 2], label = 'Class 2', color = 'blue')
    plt.scatter(X_train[1][Y_train == 3], X_train[2][Y_train == 3], label = 'Class 3', color = 'green')
    plt.legend()
    plt.title('Dimensions: [2,3], $J_3$ = ' + str(np.round(float(J3_score(np.delete(X_train, 0, 0), Y_train)), 5)))
    plt.show()
    plt.clf()

    print("The difference between dimensions 1,2 and 3d is: ",
        np.round(float(J3_score(X_train, Y_train))-float(J3_score(np.delete(X_train, 2, 0), Y_train)), 5), 
        "which is so small that we can get rid of dimensin 3 without losing much information.")

class Gaussian:
    def __init__(self, X_train, Y_train):
        """Gaussian classifyer for 3 classes"""
        self.X_train = X_train
        self.Y_train = Y_train
        self.accuracy = 0

    def mean_vector(self):
        """Returns the mean vector for each class"""
        mean_1 = mean_vector(X_train, Y_train, 1)
        mean_2 = mean_vector(X_train, Y_train, 2)
        mean_3 = mean_vector(X_train, Y_train, 3)
        return mean_1, mean_2, mean_3

    def covariance_matrix(self):
        """Returns the covariance matrix for each class"""
        cov_1 = np.cov(np.array((X_train[:,Y_train == 1])))
        cov_2 = np.cov(np.array((X_train[:,Y_train == 2])))
        cov_3 = np.cov(np.array((X_train[:,Y_train == 3])))
        return cov_1, cov_2, cov_3

    def pdf(self, x):
        """
        Calculates the probability density function for each class
        
        Parameters
        ----------
        - x: The set of feature vectors
        
        Returns
        -------
        - p1, p2, p3: The probability density function for each class
        """

        # calculating the mean vector and covariance matrix for each class
        mean_1, mean_2, mean_3 = self.mean_vector()
        cov_1, cov_2, cov_3 = self.covariance_matrix()

        # calculating the probability for each class (using the multivariate gaussian distribution)
        denominator = (2*np.pi)**(len(mean_1)/2)*np.linalg.det(cov_1)**(1/2)
        exponent = -0.5*(x-mean_1).T@np.linalg.inv(cov_1)@(x-mean_1)
        p1 = np.diag(np.exp(exponent))/denominator # note: we take the diagonal since the other values don't matter for us

        denominator = (2*np.pi)**(len(mean_2)/2)*np.linalg.det(cov_2)**(1/2)
        exponent = -0.5*(x-mean_2).T@np.linalg.inv(cov_2)@(x-mean_2)
        p2 = np.diag(np.exp(exponent))/denominator

        denominator = (2*np.pi)**(len(mean_3)/2)*np.linalg.det(cov_3)**(1/2)
        exponent = -0.5*(x-mean_3).T@np.linalg.inv(cov_3)@(x-mean_3)
        p3 = np.diag(np.exp(exponent))/denominator

        return p1, p2, p3

    def predict(self, X, Y, accuracy = True):
        """
        Predicts the class for each feature vector in X
        
        Parameters
        ----------
        - X: The set of feature vectors
        - Y: Known labels for the feature vectors
        - accuracy: If True, the accuracy of the prediction is calculated and stored in self.accuracy
        
        Returns
        -------
        - pred: The predicted class for each feature vector in X
        """
        p1, p2, p3 = self.pdf(X)
        pred = np.argmax(np.array([p1, p2, p3]), axis = 0) + 1
        if accuracy == True:
            self.accuracy = np.sum(pred == Y)/len(Y)
        return pred

    def plot(self, X, Y, title, resolution = 100):
        """
        Plots the dataset and the decision boundary
        
        Parameters
        ----------
        - X: The set of feature vectors
        - Y: Known labels for the feature vectors
        - title: The title of the plot (Gaussian classifier for "title")
        - resolution: The resolution of the plot
        """
        x, y = np.linspace(-4.0, 4.2, num=resolution), np.linspace(-3.7, 4.2, num=resolution)
        XY = np.asarray(np.meshgrid(x,y)).reshape(2, -1)

        pred = self.predict(XY, Y, accuracy = False)

        plt.pcolormesh(x, y, pred.reshape(resolution,resolution), cmap = 'Set1', alpha=0.6, vmin=0, vmax=12)
        plt.scatter(X[0][Y == 1], X[1][Y == 1],marker='+', label = 'Class 1', color = 'red')
        plt.scatter(X[0][Y == 2], X[1][Y == 2],marker='+', label = 'Class 2', color = 'blue')
        plt.scatter(X[0][Y == 3], X[1][Y == 3],marker='+', label = 'Class 3', color = 'green')
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.legend()
        plt.title('Gaussian classifier:' + title)
        plt.show()

    def plot_confusionmatrix(self, X, Y, title):
        pred = self.predict(X, Y, accuracy = False)
        confusion_matrix = metrics.confusion_matrix(Y, pred)
        confusion_matrix_display = metrics.ConfusionMatrixDisplay(confusion_matrix, display_labels = [1,2,3])
        confusion_matrix_display.plot(cmap="Reds")
        plt.title("Gaussian classifier: " + title)
        plt.grid(False)
        plt.show()
        plt.clf()

class Parzen:
    def __init__(self, X_train, Y_train):
        """Parzen window density estimator for 3 classes"""
        self.X_train = X_train
        self.Y_train = Y_train
        self.accuracy = 0

    def pdf(self, x, h):
        """
        Calculates the probability density function for each class

        Parameters
        ----------
        - x: The set of feature vectors
        - h: The window size

        Returns
        -------
        - p1, p2, p3: The probability density function for each class
        """
        
        # separating the training data into the respective classes
        X1 = self.X_train[:,Y_train == 1]
        X2 = self.X_train[:,Y_train == 2]
        X3 = self.X_train[:,Y_train == 3]

        N = len(X1[0]) # number of feature vectors in each class
        l = len(X1) # number of dimensions

        denominator = ((2*np.pi)**(l/2))*h**l # denominator of the probability density function
        p1, p2, p3 = np.zeros(len(x[0])), np.zeros(len(x[0])), np.zeros(len(x[0])) # initializing empty arrays
        for i in range(N):
            exponent1 = np.diag(-((x.T-X1[:,i]) @ (x.T-X1[:,i]).T))/(2*h**2) # exponent of the probability density function for class 1
            p1 += np.exp(exponent1)/(denominator*N) # adding the probability density function for each feature vector

            exponent2 = np.diag(-((x.T-X2[:,i]) @ (x.T-X2[:,i]).T)/(2*h**2))
            p2 += np.exp(exponent2)/(denominator*N)

            exponent3 = np.diag(-((x.T-X3[:,i]) @ (x.T-X3[:,i]).T)/(2*h**2))
            p3 += np.exp(exponent3)/(denominator*N)
        
        return p1, p2, p3

    def predict(self, X, Y, h, accuracy = True):
        """
        Predicts the class for each feature vector in X

        Parameters
        ----------
        - X: The set of feature vectors
        - Y: Known labels for the feature vectors
        - h: The window size
        - accuracy: If True, the accuracy of the prediction is calculated and stored in self.accuracy

        Returns
        -------
        - pred: The predicted class for each feature vector in X
        """

        p1, p2, p3 = self.pdf(X, h) # getting the pdfs
        pred = np.argmax(np.array([p1, p2, p3]), axis = 0) + 1 # predicting the class for each feature vector
        if accuracy == True:
            self.accuracy = np.sum(pred == Y)/len(Y) # calculating the accuracy
        return pred

    def plot(self, X, Y, h, title, resolution = 50):
        """
        Plots the dataset and the decision boundary

        Parameters
        ----------
        - X: The set of feature vectors
        - Y: Known labels for the feature vectors
        - h: The window size
        - title: The title of the plot (Parzen window density estimator for "title", h = "h")
        - resolution: The resolution of the plot

        Returns
        -------
        - pred: The predicted class for each feature vector in X
        """

        x, y = np.linspace(-4.0, 4.2, num=resolution), np.linspace(-3.7, 4.2, num=resolution) # creating the grid
        XY = np.asarray(np.meshgrid(x,y)).reshape(2, -1)

        pred = self.predict(XY, Y, h, accuracy = False) # predicting the class for each point on the grid

        plt.pcolormesh(x, y, pred.reshape(resolution,resolution), cmap = 'Set1', alpha=0.6, vmin=0, vmax=12)
        plt.scatter(X[0][Y == 1], X[1][Y == 1], marker = '+', label = 'Class 1', color = 'red')
        plt.scatter(X[0][Y == 2], X[1][Y == 2], marker = '+', label = 'Class 2', color = 'blue')
        plt.scatter(X[0][Y == 3], X[1][Y == 3], marker = '+', label = 'Class 3', color = 'green')
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.legend()
        plt.title('Parzen classifier:' + title + ', h = ' + str(h))
        plt.show()

    def plot_confusionmatrix(self, X, Y, h, title):
        """
        Plots the confusion matrix
        
        Parameters
        ----------
        - X: The set of feature vectors
        - Y: Known labels for the feature vectors
        - h: The window size
        - title: The title of the plot (Confusion matrix for "title", h = "h")
        """
        pred = self.predict(X, Y, h, accuracy = False)
        confusion_matrix = metrics.confusion_matrix(Y, pred)
        confusion_matrix_display = metrics.ConfusionMatrixDisplay(confusion_matrix, display_labels = [1,2,3])
        confusion_matrix_display.plot(cmap="Reds")
        plt.title("Parzen classifier: " + title + ", h = " + str(h))
        plt.grid(False)
        plt.show()

if __name__ == "__main__":
    # task 1
    print_J3() # printing the J3 scores
    plot_data() # plotting the training data

    # removing dimension 3 from the training and test sets
    X_train = np.delete(X_train, 2, 0)
    X_test = np.delete(X_test, 2, 0)

    # task 2
    print("------------Task 2------------")

    # Gaussian classifier
    # plotting the results
    gaussian = Gaussian(X_train, Y_train)
    gaussian.plot(X_train, Y_train, 'Training set')
    gaussian.plot(X_test, Y_test, ' Test set')
    gaussian.plot_confusionmatrix(X_test, Y_test, 'Test set')

    # printing the results
    table = PrettyTable()
    table.title = "Gaussian classifier"
    table.field_names = ["Data set", "Accuracy", "Misclassifications"]
    gaussian.predict(X_train, Y_train, accuracy = True)
    table.add_row(["Training set", np.round(gaussian.accuracy, 3)*100, np.round(len(Y_train)*(1-gaussian.accuracy), 3)])
    gaussian.predict(X_test, Y_test, accuracy = True)
    table.add_row(["Test set", np.round(gaussian.accuracy, 3)*100, np.round(len(Y_test)*(1-gaussian.accuracy), 3)])
    print(table)

    # Parzen classifyer
    h = 0.095 # optimal (found by trial and error)
    #h = 0.001 # overfitting (extreme case)

    # plotting the results
    parzen = Parzen(X_train, Y_train)
    parzen.plot(X_test, Y_test, h, ' Test set')
    parzen.plot_confusionmatrix(X_test, Y_test, h, 'Test set')

    # printing the results
    table = PrettyTable()
    table.title = "Parzen classifier (h = " + str(h) + ")"
    table.field_names = ["Data set", "Accuracy", "Misclassifications"]
    parzen.predict(X_train, Y_train, h, accuracy = True)
    table.add_row(["Training set", np.round(parzen.accuracy, 3)*100, np.round(len(Y_train)*(1-parzen.accuracy), 3)])
    parzen.predict(X_test, Y_test, h, accuracy = True)
    table.add_row(["Test set", np.round(parzen.accuracy, 3)*100, np.round(len(Y_test)*(1-parzen.accuracy), 3)])
    print(table)