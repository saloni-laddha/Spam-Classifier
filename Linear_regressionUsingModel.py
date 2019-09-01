from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

#providing path and data to system
path="C:/Users/Shiva Umesh Varun/Desktop/Python/"
Xtr=np.load(path+"Xtrain.npy")
Ytr=np.load(path+"Ytrain.npy")
Xts=np.load(path+"Xtest.npy")
Yts=np.load(path+"Ytest.npy")# Model Intialization
reg = LinearRegression()
# Data Fitting
reg = reg.fit(Xtr, Ytr)
# Y Prediction
Y_pred = reg.predict(Xtr)

rmse = np.sqrt(mean_squared_error(Ytr, Y_pred))
r2 = reg.score(Xtr, Ytr)

print(rmse)
print(r2)