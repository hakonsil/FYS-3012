import numpy as np
import matplotlib.pyplot as plt

# Defining the given variables
mu_1 = np.array([[1, 1]])
mu_2 = np.array([[1.5, 1.5]])
sigma = np.array([[0.2,0],[0,0.2]])
L = np.array([[0, 1], [0.5, 0]])
P = 1/2


def p(x, mu, sigma2):
    '''
    This function gives the probability for each class from the pdf given in the task

    We're taking the diagonal of the diagonal of the exponent of the exponent here.
    We do this as the output of this exponent would be a NxN sized matrix, where 
    N is the number of input points. We're only interested in the case of 
    (x_i - mu)@S^⁻1 @(x_i - mu).T, so therefore the diagonal. 
    '''
    
    exponent = -0.5 * (x - mu) @ np.linalg.inv(sigma2) @ (x - mu).T
    denominator = (2 * np.pi) ** (len(mu) / 2) * np.sqrt(np.linalg.det(sigma2))
    
    return np.diag(np.exp(exponent))/denominator

# creating a meshgrid of points to evaluate the pdf on
x, y = np.linspace(-0.5, 2.5, num=100), np.linspace(-0.5, 2.5, num=100)
# reshaping the meshgrid to a 2xN array, where N is the number of points
XY = np.asarray(np.meshgrid(x,y)).reshape(2, -1).T 

# calculating the probability for each class at each of the N points
p1 = p(XY, mu_1, sigma)
p2 = p(XY, mu_2, sigma)


# creating a 2xN array of predictions, for each point we can compare the two pdfs to see where the probability is higher
# from bayes decision rule we know to classify the point into the class for which the probability is higher
pred = np.array([p1, p2])

"""
The following plot shows probability of each point belonging to class to which
they are more likely to belong to? 
We can sort of see where the decision boundary would be.
"""
boundary = np.max(pred, axis=0).reshape(100,100)
X, Y = np.meshgrid(x,y)

plt.figure()
plt.contourf(X, Y, boundary)
plt.scatter(mu_1[0,0], mu_1[0,1], c='r', label='Mu1')
plt.scatter(mu_2[0,0], mu_2[0,1], c='g', label='Mu2')
plt.legend()
plt.colorbar()
plt.title('No risk contour plot')
plt.show()

# Plotting the decision boundary

# by taking the argmax we split the points into two groups
# by index (0,1) where 0 is class 1 and 1 is class 2
boundary = np.argmax(pred, axis=0).reshape(100,100)
X, Y = np.meshgrid(x,y)

plt.figure()
plt.contourf(X, Y, boundary)
plt.scatter(mu_1[0,0], mu_1[0,1], c='r', label='Mu1')
plt.scatter(mu_2[0,0], mu_2[0,1], c='g', label='Mu2')
plt.legend()
plt.title('No risk decision boundary')
plt.show()

"""Now plotting with risk"""
#This risk factor is taken from p.18 in the book
risk_factor = L[0,1]/L[1,0]

# this part is the exact same as earlier, only difference is the risk factor
pred = np.array([p1*risk_factor, p2])
boundary = np.max(pred, axis=0).reshape(100,100)
X, Y = np.meshgrid(x,y)

plt.figure()
plt.contourf(X, Y, boundary)
plt.scatter(mu_1[0,0], mu_1[0,1], c='r', label='Mu1')
plt.scatter(mu_2[0,0], mu_2[0,1], c='g', label='Mu2')
plt.colorbar()
plt.legend()
plt.title('Risk contour plot')
plt.show()

# Plotting the decision boundary
boundary = np.argmax(pred, axis=0).reshape(100,100)
X, Y = np.meshgrid(x,y)

plt.figure()
plt.contourf(X, Y, boundary)
plt.scatter(mu_1[0,0], mu_1[0,1], c='r', label='Mu1')
plt.scatter(mu_2[0,0], mu_2[0,1], c='g', label='Mu2')
plt.colorbar()
plt.legend()
plt.title('Risk decision boundary')
plt.show()

"""Classifying random feature vectors"""
# Generating 100 random feature vectors from each class
X1 = np.random.multivariate_normal(mean=mu_1.flatten(), cov=sigma, size=(100))
X2 = np.random.multivariate_normal(mean=mu_2.flatten(), cov=sigma, size=(100))
X = np.concatenate((X1, X2), axis=0)

# Generating the ground truth to compare our predictions to
Y1 = np.zeros(100)
Y2 = np.ones(100)
Y = np.concatenate((Y1, Y2), axis=0)

# The risk factor is the same as before
risk_factor = L[0,1]/L[1,0]

# minimum error classifyer

# Calculating the probability for each class for each feature vector
p1, p2 = p(X, mu_1, sigma), p(X, mu_2, sigma)
pred = np.array([p1, p2])
pred = np.argmax(pred, axis=0)

# calculating the accuracy by comparing the predictions to the ground truth
accuracy = np.sum(pred == Y)/len(pred)

print(f'The accuracy of our model is {accuracy}')

# plotting the result
x, y = np.linspace(-0.5, 3.0, num=100), np.linspace(-0.5, 3.5, num=100)
XY = np.asarray(np.meshgrid(x,y)).reshape(2, -1).T 

p1 = p(XY, mu_1, sigma).reshape(100,100)
p2 = p(XY, mu_2, sigma).reshape(100,100)

#Going of bayes decision rule, we assign a point x to class 1 if p1 > p2, and vice versa
pred = np.array([p1, p2])
boundary = np.argmax(pred, axis=0)
X, Y = np.meshgrid(x,y)

plt.figure()
plt.contourf(X, Y, boundary)

plt.scatter(X1[:,0], X1[:,1], c='r')
plt.scatter(X2[:,0], X2[:,1], c='g')

plt.scatter(mu_1[0,0], mu_1[0,1], c='k',marker='x', label='Mu1')
plt.scatter(mu_2[0,0], mu_2[0,1], c='b',marker='x', label='Mu2')
plt.legend()
plt.title('Minimum error classifyer')
plt.show()

# now on to the minimum risk classifyer

# Generating 100 random feature vectors from each class
X1 = np.random.multivariate_normal(mean=mu_1.flatten(), cov=sigma, size=(100))
X2 = np.random.multivariate_normal(mean=mu_2.flatten(), cov=sigma, size=(100))
X = np.concatenate((X1, X2), axis=0)

# Generating the ground truth to compare our predictions to
Y1 = np.zeros(100)
Y2 = np.ones(100)
Y = np.concatenate((Y1, Y2), axis=0)
# Calculating the probability for each class for each feature vector
p1, p2 = p(X, mu_1, sigma), p(X, mu_2, sigma)
pred = np.array([p1*risk_factor, p2])
pred = np.argmax(pred, axis=0)

# calculating the accuracy by comparing the predictions to the ground truth
accuracy = np.sum(pred == Y)/len(pred)

print(f'The accuracy of our model is {accuracy}')

# plotting the result
x, y = np.linspace(-0.5, 3.0, num=100), np.linspace(-0.5, 3.5, num=100)
XY = np.asarray(np.meshgrid(x,y)).reshape(2, -1).T 

p1 = p(XY, mu_1, sigma).reshape(100,100)
p2 = p(XY, mu_2, sigma).reshape(100,100)

#Going of bayes decision rule, we assign a point x to class 1 if p1 > p2, and vice versa
pred = np.array([p1*risk_factor, p2])
boundary = np.argmax(pred, axis=0)
X, Y = np.meshgrid(x,y)

plt.figure()
plt.contourf(X, Y, boundary)

plt.scatter(X1[:,0], X1[:,1], c='r')
plt.scatter(X2[:,0], X2[:,1], c='g')

plt.scatter(mu_1[0,0], mu_1[0,1], c='k',marker='x', label='Mu1')
plt.scatter(mu_2[0,0], mu_2[0,1], c='b',marker='x', label='Mu2')
plt.legend()
plt.title('Minimum risk classifyer')
plt.show()


"""now repeating the same thing with a new mu_2"""
mu_2 = np.array([[3.0, 3.0]])
print("The following results are for mu_2 = [3.0, 3.0]")
# Generating 100 random feature vectors from each class
X1 = np.random.multivariate_normal(mean=mu_1.flatten(), cov=sigma, size=(100))
X2 = np.random.multivariate_normal(mean=mu_2.flatten(), cov=sigma, size=(100))
X = np.concatenate((X1, X2), axis=0)

# Generating the ground truth to compare our predictions to
Y1 = np.zeros(100)
Y2 = np.ones(100)
Y = np.concatenate((Y1, Y2), axis=0)

# The risk factor is the same as before
risk_factor = L[0,1]/L[1,0]

# minimum error classifyer

# Calculating the probability for each class for each feature vector
p1, p2 = p(X, mu_1, sigma), p(X, mu_2, sigma)
pred = np.array([p1, p2])
pred = np.argmax(pred, axis=0)

# calculating the accuracy by comparing the predictions to the ground truth
accuracy = np.sum(pred == Y)/len(pred)

print(f'The accuracy of our model is {accuracy}')

# plotting the result
x, y = np.linspace(-0.5, 5.0, num=100), np.linspace(-0.5, 5, num=100)
XY = np.asarray(np.meshgrid(x,y)).reshape(2, -1).T 

p1 = p(XY, mu_1, sigma).reshape(100,100)
p2 = p(XY, mu_2, sigma).reshape(100,100)

#Going of bayes decision rule, we assign a point x to class 1 if p1 > p2, and vice versa
pred = np.array([p1, p2])
boundary = np.argmax(pred, axis=0)
X, Y = np.meshgrid(x,y)

plt.figure()
plt.contourf(X, Y, boundary)

plt.scatter(X1[:,0], X1[:,1], c='r')
plt.scatter(X2[:,0], X2[:,1], c='g')

plt.scatter(mu_1[0,0], mu_1[0,1], c='k',marker='x', label='Mu1')
plt.scatter(mu_2[0,0], mu_2[0,1], c='b',marker='x', label='Mu2')
plt.legend()
plt.title('Minimum error classifyer')
plt.show()

# now on to the minimum risk classifyer

# Generating 100 random feature vectors from each class
X1 = np.random.multivariate_normal(mean=mu_1.flatten(), cov=sigma, size=(100))
X2 = np.random.multivariate_normal(mean=mu_2.flatten(), cov=sigma, size=(100))
X = np.concatenate((X1, X2), axis=0)

# Generating the ground truth to compare our predictions to
Y1 = np.zeros(100)
Y2 = np.ones(100)
Y = np.concatenate((Y1, Y2), axis=0)
# Calculating the probability for each class for each feature vector
p1, p2 = p(X, mu_1, sigma), p(X, mu_2, sigma)
pred = np.array([p1*risk_factor, p2])
pred = np.argmax(pred, axis=0)

# calculating the accuracy by comparing the predictions to the ground truth
accuracy = np.sum(pred == Y)/len(pred)

print(f'The accuracy of our model is {accuracy}')

# plotting the result
x, y = np.linspace(-0.5, 5.0, num=100), np.linspace(-0.5, 5, num=100)
XY = np.asarray(np.meshgrid(x,y)).reshape(2, -1).T 

p1 = p(XY, mu_1, sigma).reshape(100,100)
p2 = p(XY, mu_2, sigma).reshape(100,100)

#Going of bayes decision rule, we assign a point x to class 1 if p1 > p2, and vice versa
pred = np.array([p1*risk_factor, p2])
boundary = np.argmax(pred, axis=0)
X, Y = np.meshgrid(x,y)

plt.figure()
plt.contourf(X, Y, boundary)

plt.scatter(X1[:,0], X1[:,1], c='r')
plt.scatter(X2[:,0], X2[:,1], c='g')

plt.scatter(mu_1[0,0], mu_1[0,1], c='k',marker='x', label='Mu1')
plt.scatter(mu_2[0,0], mu_2[0,1], c='b',marker='x', label='Mu2')
plt.legend()
plt.title('Minimum risk classifyer')
plt.show()