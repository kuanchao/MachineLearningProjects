# SVM Regression
import numpy as np
from sklearn import datasets
from sklearn.svm import SVR
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC

# load the datasets
##dataset = datasets.load_diabetes()
# fit a SVM model to the data

import pandas as pd
mydata = pd.read_csv('winequality-red.csv')
dataset = mydata
dataset.target = mydata["quality"]
#provided your csv has header row, and the label column is named "Label"

#select all but the last column as data
dataset.data = mydata.ix[:,:-1]
model = SVR()
model.fit(dataset.data, dataset.target)
print(model)
# make predictions
expected = dataset.target
predicted = model.predict(dataset.data)
# summarize the fit of the model
mse = np.mean((predicted-expected)**2)
print(mse)
print(model.score(dataset.data, dataset.target))

train_sizes, train_scores, valid_scores = learning_curve(SVC(kernel='linear'),
dataset.data, dataset.target, train_sizes=[100, 200, 300, 400, 500, 600, 700, 800], cv=5)
train_sizes
print (train_scores)
print (valid_scores)
