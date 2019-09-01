import numpy as np 
import sys
from sklearn.neighbors import KNeighborsClassifier

#providing path and data to system
path="C:/Users/Shiva Umesh Varun/Desktop/Python/"
Xtr=np.load(path+"Xtrain.npy")
Ytr=np.load(path+"Ytrain.npy")
Xts=np.load(path+"Xtest.npy")
Yts=np.load(path+"Ytest.npy")
print(Xtr.shape, Xts.shape)


#start working
classifier=KNeighborsClassifier(n_neighbors=1)
classifier.fit(Xtr, Ytr)
predictions=classifier.predict(Xts)
accuracy = 0.0
for i in range(len(predictions)):
    if predictions[i] == Yts[i]:
        accuracy += 1
accuracy /= len(predictions)
accuracy *= 100
test_accuracy = accuracy
print(test_accuracy)