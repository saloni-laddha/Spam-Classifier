import numpy as np 
import sys
class logistic_classifier():
    def __init__(self):
        self.w=0.0
        pass



    def sigmoid(self,z):
        for i in range(len(z)):
            if z[i]<-5:
                z[i]=-3
            elif z[i]>5:
                z[i]=3
            else:pass
        return 1/(1+np.exp(-z))


    def compute_loss(self,probabilities,Ytr):
        for i in range(len(probabilities)):
            if probabilities[i]<0.00001:
                probabilities[i]=probabilities[i]+0.00001
        return (-Ytr * np.log(probabilities) - (1 - Ytr) * np.log(1 - probabilities))


    def compute_probabilities(self,Xtr):
        z=np.dot(Xtr,self.w)
        return self.sigmoid(z)




    def fit(self,Xtr,Ytr):
        '''
        This function trains the logistic regression model on the 
        given training data
        '''
        # num_iters: number of iterations that gradient descent should run for
        learning_rate=0.00005
        num_iters=10000
        self.w=np.random.normal(0.0,0.1,Xtr.shape[1])
        self.w=np.zeros(Xtr.shape[1])

        for iter in range(num_iters):
            probabilities=self.compute_probabilities(Xtr)
            gradient=np.dot(Xtr.T,(probabilities-Ytr))/Ytr.size
            self.w=self.w-learning_rate*gradient
            train_loss = self.compute_loss(probabilities, Ytr)

    def predict(self,Xts):
        linear_combinations = np.matmul(Xts, self.w)
        probabilities = self.sigmoid(linear_combinations)
        self.predictions = np.zeros(probabilities.shape)
        self.predictions = (probabilities > 0.5)
        return self.predictions

# adjust the path according to where you have stored the dataset. 
path = "C:/Users/Shiva Umesh Varun/Desktop/Python/"
Xtr = np.load(path + "Xtrain.npy")
Ytr = np.load(path + "Ytrain.npy")
Xts = np.load(path + "Xtest.npy")
Yts = np.load(path + "Ytest.npy")
model=logistic_classifier()
model.fit(Xtr,Ytr)
predictions=model.predict(Xts)
accuracy = 0.0
for i in range(len(predictions)):
    if predictions[i] == Yts[i]:
        accuracy += 1
accuracy /= len(predictions)
accuracy *= 100
test_accuracy = accuracy
print(test_accuracy)