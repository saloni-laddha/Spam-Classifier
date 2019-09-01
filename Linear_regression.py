import numpy as np 
import sys
#Class for linear regression
class Linear_regression():
    def __init__(self):
        pass
    #fit data to class
    def fit(self,Xtr,Ytr):
        self.Xtr=Xtr
        self.Ytr=Ytr
    #function to calculate cost
    def cost_function(self,Xtr,Ytr,B):
        m=len(Ytr)
        j=np.sum((Xtr.dot(B)-Ytr)**2)/(2*m)
        return j
    def gradient_descent(self,Xtr,Ytr,B,alpha,iterations):
        print(self.cost_function(Xtr,Ytr,B))
        m=len(Ytr)
        for iterations in range(100000):
            h=Xtr.dot(B)
            loss=h-Ytr
            gradient=loss.T.dot(Xtr)/m
            B=B-gradient*alpha
        return B
#providing path and data to system
path="C:/Users/Shiva Umesh Varun/Desktop/Python/"
Xtr=np.load(path+"Xtrain.npy")
Ytr=np.load(path+"Ytrain.npy")
Xts=np.load(path+"Xtest.npy")
Yts=np.load(path+"Ytest.npy")

#initial value for B
B=np.zeros(Xtr.shape[1])
Linear_regress=Linear_regression()
Linear_regress.fit(Xtr,Ytr)
B=Linear_regress.gradient_descent(Xtr,Ytr,B,0.000001,10000)
print(B)
print(Linear_regress.cost_function(Xtr,Ytr,B))