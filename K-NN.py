import numpy as np 
import sys

#class for K-Neighbour
class K_nearest():
	def __init__(self,k):
		self.k=k
	def fit(self,Xtr,Ytr):
		self.Xtr=Xtr
		self.Ytr=Ytr
	#Function to predict the k-nearest neighbour
	def predict(self,Xts):

		Y_New_value=np.zeros(Xts.shape[0]) #variable to store regression of test data
		for i in range(len(Xts)):
			#this loop will run for all test data
			distance=np.zeros(self.Xtr.shape[0])
			for j in range(len(Xtr)):
				#for each of test data,this loop will calculate the distance with each training set
				distance[j]=np.linalg.norm(Xts[i]-Xtr[j]) #calculating distance

			distance_sort=np.argsort(distance) #function will give the indices according to sorted function
			num_positive=0
			for k in range(self.k):
				#getting nearest value
				num_positive+=self.Ytr[distance_sort[k]]
				#checking majority
			if num_positive>self.k/2:
				Y_New_value[i]=1
			else:
				Y_New_value[i]=0
		return Y_New_value

#providing path and data to system
path="C:/Users/Shiva Umesh Varun/Desktop/Python/"
Xtr=np.load(path+"Xtrain.npy")
Ytr=np.load(path+"Ytrain.npy")
Xts=np.load(path+"Xtest.npy")
Yts=np.load(path+"Ytest.npy")
print(Xtr.shape, Xts.shape)


#start working
k=int(input("give value of k"),10)
classifier = K_nearest(k)
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