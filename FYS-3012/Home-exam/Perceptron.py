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
rho = 0.001 # learning rate
T = 4 # number of epochs
wa_0 = np.random.rand(3,1) # random initial weights
wb_0 = np.random.rand(3,1)
wc_0 = np.random.rand(3,1)
w_0 = np.asarray([wa_0, wb_0, wc_0])


"""wa_0 = np.asarray(([0.70408672],[0.71747854], [0.75285463]))
wb_0 = np.asarray(([0.27109956],[0.1118474 ], [0.54355472]))
wc_0 = np.asarray(([0.78227421],[0.08976388], [0.87141951]))
w_0 = np.asarray([wa_0, wb_0, wc_0])
"""


# defining the activation function
def activation(x, a=1):
    f = 1/(1+np.exp(-a*x))
    return f

def derivative_activation(x, a=1):
    f = a*activation(x)*(1-activation(x))
    return f

# discrimination function
def discriminant(x, w):
    f = w.T @ x
    return f

def delta_w(x, y, y_hat, w, rho=rho):
    f1 = rho*(y-y_hat)*derivative_activation(discriminant(x[0], w[0]))*x[0]
    f2 = rho*(y-y_hat)*derivative_activation(discriminant(x[1], w[1]))*x[1]
    f3 = rho*(y-y_hat)*derivative_activation(discriminant(x[2], w[2]))*x[2]
    f = np.asarray([f1, f2, f3]).reshape(3,1)
    return f

def cycle(X_train, Y_train, w_0, rho=rho, T=T):
    w = w_0
    for t in range(T):
        #print("t=",t+1)
        dw = np.zeros((3,1))
        for i in range(len(X_train[0])):
            x = np.matrix(X_train[:,i]).T
            y = Y_train[i]

            y_hat = activation(discriminant(x, w_0))

            dw += delta_w(x, y, y_hat, w, rho=rho,)

        w += dw
    return w

def perceptron(X_test, w):
    y_hat = np.zeros(len(X_test[0]))
    for i in range(len(X_test[0])):
        x = np.matrix((1, X_test[0,i], X_test[1,i])).T
        y_hat[i] = activation(discriminant(x, w))
    return y_hat

def perceptron_network(X_train, Y_train, X_test, w_0, rho=rho, T=T):
    Y_train_class_1 = np.zeros(len(Y_train))
    Y_train_class_2 = np.zeros(len(Y_train))
    Y_train_class_3 = np.zeros(len(Y_train))

    Y_train_class_1[Y_train == 1] = 1
    Y_train_class_2[Y_train == 2] = 1
    Y_train_class_3[Y_train == 3] = 1

    w_1 = cycle(X_train, Y_train_class_1, w_0[0], rho=rho, T=T)
    w_2 = cycle(X_train, Y_train_class_2, w_0[1], rho=rho, T=T)
    w_3 = cycle(X_train, Y_train_class_3, w_0[2], rho=rho, T=T)

    output_1 = perceptron(X_test, w_1)
    output_2 = perceptron(X_test, w_2)
    output_3 = perceptron(X_test, w_3)

    pred = np.argmax(np.asarray((output_1, output_2, output_3)), axis=0)+1
    return pred, w_1, w_2, w_3

pred_train, w1, w2, w3 = perceptron_network(X_train, Y_train, np.delete(X_train, 0, 0), w_0, rho=rho, T=T)
accuracy_train = np.sum(pred_train == Y_train)/len(Y_train)
print("Training set accuracy: ", accuracy_train)

output_1, output_2, output_3 = perceptron(X_test, w1), perceptron(X_test, w2), perceptron(X_test, w3)
pred_test = np.argmax(np.asarray((output_1, output_2, output_3)), axis=0)+1

accuracy_test = np.sum(pred_test == Y_test)/len(Y_test)
print("Test set accuracy: ", accuracy_test)
print("T = ", T, ", rho = ", rho)

resolution = 100
x, y = np.linspace(-4.0, 3.7, num=resolution), np.linspace(-3.0, 4.2, num=resolution)
XY = np.asarray(np.meshgrid(x,y)).reshape(2, -1)

output_1, output_2, output_3 = perceptron(XY, w1), perceptron(XY, w2), perceptron(XY, w3)
pred = np.argmax(np.asarray((output_1, output_2, output_3)), axis=0)+1

plt.pcolormesh(x, y, pred.reshape(resolution, resolution), cmap = 'Set1', alpha=0.6, vmin=0, vmax=12)
plt.scatter(X_test[0, Y_train == 1], X_test[1, Y_train == 1], color='red', label='Class 1')
plt.scatter(X_test[0, Y_train == 2], X_test[1, Y_train == 2], color='blue', label='Class 2')
plt.scatter(X_test[0, Y_train == 3], X_test[1, Y_train == 3], color='green', label='Class 3')
plt.xlim(-4, 3.7)
plt.ylim(-3, 4.2)
plt.legend()
plt.show()

t = np.linspace(1, T, num=T)
train_accuracy = np.zeros(T)
test_accuracy = np.zeros(T)
for i in range(T):
    pred_train, w1, w2, w3 = perceptron_network(X_train, Y_train, np.delete(X_train, 0, 0), w_0, rho=rho, T=int(t[i]))
    accuracy_train = np.sum(pred_train == Y_train)/len(Y_train)
    train_accuracy[i] = accuracy_train

    output_1, output_2, output_3 = perceptron(X_test, w1), perceptron(X_test, w2), perceptron(X_test, w3)
    pred_test = np.argmax(np.asarray((output_1, output_2, output_3)), axis=0)+1
    accuracy_test = np.sum(pred_test == Y_test)/len(Y_test)
    test_accuracy[i] = accuracy_test

output_1, output_2, output_3 = perceptron(X_train, wa_0), perceptron(X_train, wb_0), perceptron(X_train, wc_0)
pred_train = np.argmax(np.asarray((output_1, output_2, output_3)), axis=0)+1
accuracy_train_0 = np.sum(pred_train == Y_train)/len(Y_train)
train_accuracy = np.concatenate((np.asarray([accuracy_train_0]), train_accuracy), axis=0)

output_1, output_2, output_3 = perceptron(X_test, wa_0), perceptron(X_test, wb_0), perceptron(X_test, wc_0)
pred_test = np.argmax(np.asarray((output_1, output_2, output_3)), axis=0)+1
accuracy_test_0 = np.sum(pred_test == Y_test)/len(Y_test)
test_accuracy = np.concatenate((np.asarray([accuracy_test_0]), test_accuracy), axis=0)

t = np.concatenate((np.asarray([0]), t), axis=0)

plt.plot(t, train_accuracy, label='Training set accuracy')
plt.plot(t, test_accuracy, label='Test set accuracy')
plt.xlabel('Number of epochs')
plt.ylabel('Accuracy')
plt.title("Accuracy of Perceptron over epochs for $rho$ = " + str(rho))
plt.legend()
plt.show()

print("Training set accuracy: ", train_accuracy[-1])
print("Test set accuracy: ", test_accuracy[-1])