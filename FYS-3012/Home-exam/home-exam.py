import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import scipy as sc

# Importing the data
dataset = sc.io.loadmat('Home-exam\ExamData3D.mat')

# Splitting the data into training and test sets
X_train = dataset['X_train']
X_test = dataset['X_test']
Y_train = dataset['Y_train'][0]
Y_test = dataset['Y_test'][0]

# defining variables
P = 1/3 # probability of each class (equiprobable classes)

"""Task 1"""

def mean_vector_3d(x: np.ndarray, y: np.ndarray, class_nr: int) -> np.ndarray:
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

    # splitting the dataset into the three dimensions
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]

    # calculating the mean for each dimension
    mu1 = np.mean(x1[y == class_nr])
    mu2 = np.mean(x2[y == class_nr])
    mu3 = np.mean(x3[y == class_nr])

    # putting the means into a vector
    mu = np.array([mu1, mu2, mu3])
    return np.matrix(mu).T

def mean_vector_2d(x: np.ndarray, y: np.ndarray, class_nr: int) -> np.ndarray:
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

    # splitting the dataset into the three dimensions
    x1 = x[0]
    x2 = x[1]

    # calculating the mean for each dimension
    mu1 = np.mean(x1[y == class_nr])
    mu2 = np.mean(x2[y == class_nr])


    # putting the means into a vector
    mu = np.array([mu1, mu2])
    return np.matrix(mu).T

def mean_vector_1d(x: np.ndarray, y: np.ndarray, class_nr: int) -> np.ndarray:
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

    # calculating the mean for each dimension
    mu1 = np.mean(x[y == class_nr])

    # putting the means into a vector
    mu = np.array([mu1])
    return np.matrix(mu).T

def J3_3d(x, y):
    # calculating the mean vector for each class
    mu1 = mean_vector_3d(x, y, 1)
    mu2 = mean_vector_3d(x, y, 2)
    mu3 = mean_vector_3d(x, y, 3)

    # calculating the global mean vector
    mu0 = np.matrix((mu1 + mu2 + mu3)*P)

    # calculating the covariance matrix for each class
    cov1 = np.cov(np.array((x[:,y == 1])))
    cov3 = np.cov(np.array((x[:,y == 3])))
    cov2 = np.cov(np.array((x[:,y == 2])))

    # calculating the within class scatter matrix
    Sw = (cov1 + cov2 + cov3)*P

    # calculating the between class scatter matrix
    Sb  = ((mu1-mu0)@((mu1-mu0).T) + (mu2-mu0)@((mu2-mu0).T) + (mu3-mu0)@((mu3-mu0).T))*P

    # calculating the modified J3 score
    J3 = np.trace(np.linalg.inv(Sw)@Sb)
    return J3

def J3_2d(x, y):
    # calculating the mean vector for each class
    mu1 = mean_vector_2d(x, y, 1)
    mu2 = mean_vector_2d(x, y, 2)
    mu3 = mean_vector_2d(x, y, 3)

    # calculating the global mean vector
    mu0 = np.matrix((mu1 + mu2 + mu3)*P)

    # calculating the covariance matrix for each class
    cov1 = np.cov(np.array((x[:,y == 1])))
    cov3 = np.cov(np.array((x[:,y == 3])))
    cov2 = np.cov(np.array((x[:,y == 2])))

    # calculating the within class scatter matrix
    Sw = (cov1 + cov2 + cov3)*P

    # calculating the between class scatter matrix
    Sb  = ((mu1-mu0)@((mu1-mu0).T) + (mu2-mu0)@((mu2-mu0).T) + (mu3-mu0)@((mu3-mu0).T))*P

    # calculating the modified J3 score
    J3 = np.trace(np.linalg.inv(Sw)@Sb)
    return J3

def J3_1d(x, y):
    # calculating the mean vector for each class
    mu1 = mean_vector_1d(x, y, 1)
    mu2 = mean_vector_1d(x, y, 2)
    mu3 = mean_vector_1d(x, y, 3)

    # calculating the global mean vector
    mu0 = np.matrix((mu1 + mu2 + mu3)*P)

    # calculating the covariance matrix for each class
    cov1 = np.cov(np.array((x[y == 1])))
    cov3 = np.cov(np.array((x[y == 3])))
    cov2 = np.cov(np.array((x[y == 2])))

    # calculating the within class scatter matrix
    Sw = (cov1 + cov2 + cov3)*P

    # calculating the between class scatter matrix
    Sb  = ((mu1-mu0)@((mu1-mu0).T) + (mu2-mu0)@((mu2-mu0).T) + (mu3-mu0)@((mu3-mu0).T))*P

    # calculating the modified J3 score
    J3 = Sb/Sw
    return J3

# printing the results

print("------------Task 1------------")
print(f"J3 score for [1]: {float(J3_1d(X_train[0], Y_train))}")
print(f"J3 score for [2]: {float(J3_1d(X_train[1], Y_train))}")
print(f"J3 score for [3]: {float(J3_1d(X_train[2], Y_train))}")
print(f"J3 score for [1,2]: {float(J3_2d(np.delete(X_train, 2, 0), Y_train))}")
print(f"J3 score for [1,3]: {float(J3_2d(np.delete(X_train, 1, 0), Y_train))}")
print(f"J3 score for [2,3]: {float(J3_2d(np.delete(X_train, 0, 0), Y_train))}")
print(f"J3 score for [1,2,3]: {float(J3_3d(X_train, Y_train))}")

# plotting the data

# plotting the 1D data
plt.hist(X_train[0][Y_train == 1], bins = 20, alpha = 0.5, label = 'Class 1', color = 'red')
plt.hist(X_train[0][Y_train == 2], bins = 20, alpha = 0.5, label = 'Class 2', color = 'blue')
plt.hist(X_train[0][Y_train == 3], bins = 20, alpha = 0.5, label = 'Class 3', color = 'green')
plt.legend()
plt.title('Dimension: [1], $J_3$ = ' + str(np.round(float(J3_1d(X_train[0], Y_train)), 5)))
plt.show()

plt.hist(X_train[1][Y_train == 1], bins = 20, alpha = 0.5, label = 'Class 1', color = 'red')
plt.hist(X_train[1][Y_train == 2], bins = 20, alpha = 0.5, label = 'Class 2', color = 'blue')
plt.hist(X_train[1][Y_train == 3], bins = 20, alpha = 0.5, label = 'Class 3', color = 'green')
plt.legend()
plt.title('Dimension: [2], $J_3$ = ' + str(np.round(float(J3_1d(X_train[1], Y_train)), 5)))
plt.show()

plt.hist(X_train[2][Y_train == 1], bins = 20, alpha = 0.5, label = 'Class 1', color = 'red')
plt.hist(X_train[2][Y_train == 2], bins = 20, alpha = 0.5, label = 'Class 2', color = 'blue')
plt.hist(X_train[2][Y_train == 3], bins = 20, alpha = 0.5, label = 'Class 3', color = 'green')
plt.legend()
plt.title('Dimension: [3], $J_3$ = ' + str(np.round(float(J3_1d(X_train[2], Y_train)), 5)))
plt.show()

# plotting the 2D data
plt.scatter(X_train[0][Y_train == 1], X_train[1][Y_train == 1], label = 'Class 1', color = 'red')
plt.scatter(X_train[0][Y_train == 2], X_train[1][Y_train == 2], label = 'Class 2', color = 'blue')
plt.scatter(X_train[0][Y_train == 3], X_train[1][Y_train == 3], label = 'Class 3', color = 'green')
plt.legend()
plt.title('Dimensions: [1,2], $J_3$ = ' + str(np.round(float(J3_2d(np.delete(X_train, 2, 0), Y_train)), 5)))
plt.show()

plt.scatter(X_train[0][Y_train == 1], X_train[2][Y_train == 1], label = 'Class 1', color = 'red')
plt.scatter(X_train[0][Y_train == 2], X_train[2][Y_train == 2], label = 'Class 2', color = 'blue')
plt.scatter(X_train[0][Y_train == 3], X_train[2][Y_train == 3], label = 'Class 3', color = 'green')
plt.legend()
plt.title('Dimensions: [1,3], $J_3$ = ' + str(np.round(float(J3_2d(np.delete(X_train, 1, 0), Y_train)), 5)))
plt.show()

plt.scatter(X_train[1][Y_train == 1], X_train[2][Y_train == 1], label = 'Class 1', color = 'red')
plt.scatter(X_train[1][Y_train == 2], X_train[2][Y_train == 2], label = 'Class 2', color = 'blue')
plt.scatter(X_train[1][Y_train == 3], X_train[2][Y_train == 3], label = 'Class 3', color = 'green')
plt.legend()
plt.title('Dimensions: [2,3], $J_3$ = ' + str(np.round(float(J3_2d(np.delete(X_train, 0, 0), Y_train)), 5)))
plt.show()

print("The difference between dimensions 1,2 and 3d is: ", float(J3_3d(X_train, Y_train))-float(J3_2d(np.delete(X_train, 2, 0), Y_train)))


#Since the value we get from including the third dimension is negligible
#we can get rid of it and only use the first two dimensions.

X_train = np.delete(X_train, 2, 0)
X_test = np.delete(X_test, 2, 0)


"""Task 2"""
print("------------Task 2------------")
"""First we have to estimate the gaussian variables"""
mean_1 = mean_vector_2d(X_train, Y_train, 1)
mean_2 = mean_vector_2d(X_train, Y_train, 2)
mean_3 = mean_vector_2d(X_train, Y_train, 3)

cov_1 = np.cov(np.array((X_train[:,Y_train == 1])))
cov_2 = np.cov(np.array((X_train[:,Y_train == 2])))
cov_3 = np.cov(np.array((X_train[:,Y_train == 3])))

# calculating the gaussian for each class
def multivariate_gaussian(x, mu, cov):
    denominator = (2*np.pi)**(len(mu)/2)*np.linalg.det(cov)**(1/2)
    exponent = -0.5*(x-mu).T@np.linalg.inv(cov)@(x-mu)
    return np.diag(np.exp(exponent))/denominator

# testing the function
p1 = multivariate_gaussian(X_train, mean_1, cov_1)
p2 = multivariate_gaussian(X_train, mean_2, cov_2)
p3 = multivariate_gaussian(X_train, mean_3, cov_3)

pred = np.argmax(np.array([p1, p2, p3]), axis = 0) + 1
accuracy = np.sum(pred == Y_train)/len(Y_train)
print("--Gaussian ML classifier--")
print(f"The accuracy for the training set is: {np.round(accuracy, 4)}")
print(f"The number of misclassified samples is: {np.sum(pred != Y_train)}")

p1 = multivariate_gaussian(X_test, mean_1, cov_1)
p2 = multivariate_gaussian(X_test, mean_2, cov_2)
p3 = multivariate_gaussian(X_test, mean_3, cov_3)

pred = np.argmax(np.array([p1, p2, p3]), axis = 0) + 1
accuracy = np.sum(pred == Y_test)/len(Y_test)
print(f"The accuracy for the test set is: {np.round(accuracy, 4)}")
print(f"The number of misclassified samples is: {np.sum(pred != Y_test)}")

resolution = 30
x, y = np.linspace(-4.0, 3.7, num=resolution), np.linspace(-3.0, 4.2, num=resolution)
XY = np.asarray(np.meshgrid(x,y)).reshape(2, -1)

p3 = multivariate_gaussian(XY, mean_3, cov_3).reshape(resolution,resolution)
p1 = multivariate_gaussian(XY, mean_1, cov_1).reshape(resolution,resolution)
p2 = multivariate_gaussian(XY, mean_2, cov_2).reshape(resolution,resolution)

pred = np.argmax(np.array([p1, p2, p3]), axis = 0) + 1

plt.pcolormesh(x, y, pred, cmap = 'Set1', alpha=0.6, vmin=0, vmax=12)
plt.scatter(X_test[0][Y_test == 1], X_test[1][Y_test == 1], label = 'Class 1', color = 'red')
plt.scatter(X_test[0][Y_test == 2], X_test[1][Y_test == 2], label = 'Class 2', color = 'blue')
plt.scatter(X_test[0][Y_test == 3], X_test[1][Y_test == 3], label = 'Class 3', color = 'green')
plt.legend()
plt.title('ML-Gaussian classifier in dimensions [1,2]')
plt.show()
plt.clf()

# creating the parzen window density estimator
def parzen(x, X_class, h):
    N = len(X_class[0])
    l = len(X_class)
    sum = np.zeros(N)
    denominator = ((2*np.pi)**(l/2))*h**l
    for i in range(N):
        x_i = np.matrix(X_class[:,i]).T
        exponent = -((x-x_i).T @ (x-x_i))/(2*h**2)
        sum[i] = np.exp(exponent)/denominator
    p = np.sum(sum)/N
    return p


h_p = 0.09
# finding the accuracy for the training set
prob1 = np.zeros(len(X_train[0]))
prob2 = np.zeros(len(X_train[0]))
prob3 = np.zeros(len(X_train[0]))

for i in range(len(X_train[0])):
    x1 = np.matrix(X_train[:,i]).T
    prob1[i] = parzen(x1, X_train[:,Y_train == 1], h_p)
    prob2[i] = parzen(x1, X_train[:,Y_train == 2], h_p)
    prob3[i] = parzen(x1, X_train[:,Y_train == 3], h_p)

pred = np.argmax(np.array([prob1, prob2, prob3]), axis = 0) + 1
accuracy = np.round(np.sum(pred == Y_train)/len(Y_train), 6)
print("--Parzen window density estimator--")
print("The accuracy for the training set is: ", accuracy)
print("The number of misclassified samples is: ", np.sum(pred != Y_train))

# finding the accuracy for the test set
prob1 = np.zeros(len(X_train[0]))
prob2 = np.zeros(len(X_train[0]))
prob3 = np.zeros(len(X_train[0]))

for i in range(len(X_test[0])):
    x1 = np.matrix(X_test[:,i]).T
    prob1[i] = parzen(x1, X_train[:,Y_train == 1], h_p)
    prob2[i] = parzen(x1, X_train[:,Y_train == 2], h_p)
    prob3[i] = parzen(x1, X_train[:,Y_train == 3], h_p)

pred = np.argmax(np.array([prob1, prob2, prob3]), axis = 0) + 1
accuracy = np.round(np.sum(pred == Y_test)/len(Y_test), 6)
print("The accuracy for the test set is: ", accuracy)
print("The number of misclassified samples is: ", np.sum(pred != Y_test))

# plotting the decision regions
resolution = 30
x, y = np.linspace(-4.0, 3.7, num=resolution), np.linspace(-3.0, 4.2, num=resolution)
XY = np.asarray(np.meshgrid(x,y)).reshape(2, -1)

prob1 = np.zeros(len(XY[0]))
prob2 = np.zeros(len(XY[0]))
prob3 = np.zeros(len(XY[0]))

for i in range(len(XY[0])):
    x1 = np.matrix(XY[:,i]).T
    prob1[i] = parzen(x1, X_train[:,Y_train == 1], h_p)
    prob2[i] = parzen(x1, X_train[:,Y_train == 2], h_p)
    prob3[i] = parzen(x1, X_train[:,Y_train == 3], h_p)

prob1 = prob1.reshape(resolution,resolution)
prob2 = prob2.reshape(resolution,resolution)
prob3 = prob3.reshape(resolution,resolution)
pred = np.argmax(np.array([prob1, prob2, prob3]), axis = 0) + 1

plt.pcolormesh(x, y, pred, cmap = 'Set1', alpha=0.6, vmin=0, vmax=12)
plt.scatter(X_test[0][Y_test == 1], X_test[1][Y_test == 1], label = 'Class 1', color = 'red')
plt.scatter(X_test[0][Y_test == 2], X_test[1][Y_test == 2], label = 'Class 2', color = 'blue')
plt.scatter(X_test[0][Y_test == 3], X_test[1][Y_test == 3], label = 'Class 3', color = 'green')
plt.legend()
plt.title('Decision region for the test set in dimensions [1,2], h = ' + str(h_p))
plt.show()
plt.clf()