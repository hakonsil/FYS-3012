import numpy as np
import matplotlib.pyplot as plt

# Defining the given variables
covariance_matrix = np.array(
                    [[1.2, 0.4]
                    ,[0.4, 1.8]])

mean_vector_1 = np.array([0.1, 0.1]).T
mean_vector_2 = np.array([2.1, 1.9]).T 
mean_vector_3 = np.array([-1.5, 2.0]).T

# Task a)
feature_vector = np.array([1.6, 1.5]).T

def discriminant_function(x, mean, sigma, l=2, p=(1/3)):
    c = -(l/2)*np.log(2*np.pi)-0.5*np.log(np.linalg.det(sigma))
    g = -0.5*x.T@np.linalg.inv(sigma)@x + 0.5*x.T@np.linalg.inv(sigma)@mean - 0.5*mean.T@np.linalg.inv(sigma)@mean + 0.5*mean.T@np.linalg.inv(sigma)@x + np.log(p) + c
    return float(g)

"""Classifying the feature vector"""
g_1 = discriminant_function(feature_vector, mean_vector_1, covariance_matrix)
g_2 = discriminant_function(feature_vector, mean_vector_2, covariance_matrix)
g_3 = discriminant_function(feature_vector, mean_vector_3, covariance_matrix)

# The feature vector is classified into the class whose discriminant function has the
# highest value at the feature vector.
if g_1 > g_2 and g_1 > g_3:
    print("The feature vector belongs to the first class.")
elif g_2 > g_1 and g_2 > g_3:
    print("The feature vector belongs to the second class.")
elif g_3 > g_1 and g_3 > g_2:
    print("The feature vector belongs to the third class.")
else:
    print("The feature vector cannot be classified.")



# Plotting the decision boundaries
"""
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z_1 = np.zeros(X.shape)
Z_2 = np.zeros(X.shape)
Z_3 = np.zeros(X.shape)
for i in range(len(X)):
    for j in range(len(X)):
        feature_vector = np.array([X[i,j], Y[i,j]]).T
        Z_1[i,j] = discriminant_function(feature_vector, mean_vector_1, covariance_matrix)
        Z_2[i,j] = discriminant_function(feature_vector, mean_vector_2, covariance_matrix)
        Z_3[i,j] = discriminant_function(feature_vector, mean_vector_3, covariance_matrix)


plt.contour(X, Y, Z_1-Z_2, levels=[0], colors='red')
plt.contour(X, Y, Z_1-Z_3, levels=[0], colors='blue')
plt.contour(X, Y, Z_2-Z_3, levels=[0], colors='green')
plt.scatter(1.6, 1.5, color='k', marker='x')
plt.title('Decision boundaries')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.show()
"""



# Task b)
# Defining the given variables
vector_1 = np.array([2.1, 1.9]).T

# calculating the mahalanobis distance
def mahalanobis_distance(x, mean, sigma):
    d = np.sqrt(((x-mean).T)@np.linalg.inv(sigma)@(x-mean))
    return float(d)

# Plotting the curves of equal mahalanobis distance
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = np.zeros(X.shape)
for i in range(len(X)):
    for j in range(len(X)):
        feature_vector = np.array([X[i,j], Y[i,j]]).T
        Z[i,j] = mahalanobis_distance(feature_vector, vector_1, covariance_matrix)

plt.contourf(X, Y, Z)
plt.colorbar()
plt.scatter(vector_1[0], vector_1[1], color='red', marker='x')
plt.title('Curves of equal Mahalanobis distance from ' + str(vector_1) + '$^T$')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
