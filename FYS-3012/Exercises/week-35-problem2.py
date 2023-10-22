import numpy as np 
from scipy.io import loadmat 

#Load data
data = loadmat('Exercises\data\Xtr_spam.mat')

xtr = data['Xtr_spam']
xte = loadmat('Exercises\data/Xte_spam.mat')['Xte_spam'].T

ytr = loadmat('Exercises\data/ytr_spam.mat')['ytr_spam'].flatten()
yte = loadmat('Exercises\data/yte_spam.mat')['yte_spam'].flatten()

print(xtr.shape, xte.shape, ytr.shape, yte.shape)
print(yte)
# convert -1 to 0?
ytr[ytr == -1] = 0
yte[yte == -1] = 0


# Compute the covariance matrix S
sigma1 = xtr[np.where(ytr==1)].std(0)
sigma2 = xtr[np.where(ytr==0)].std(0)
sigma = (sigma1+sigma2)/2

# create a diagonal covariance matrix
S = np.eye(xtr.shape[1])*sigma
print(S.shape)
