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

# defining parameters
a = np.array([1, 2, 3])
b = np.zeros(4)
b = a
print(b)