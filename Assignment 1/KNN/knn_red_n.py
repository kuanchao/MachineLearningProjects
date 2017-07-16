import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import learning_curve


# load the datasets
#dataset = datasets.load_diabetes()
# fit a model to the data
import pandas as pd
mydata = pd.read_csv('winequality-red.csv')
dataset = mydata
dataset.target = mydata["quality"]
#provided your csv has header row, and the label column is named "Label"

#select all but the last column as data
dataset.data = mydata.ix[:,:-1]
print ('uniform')
for n in range(1, 21):
    model = KNeighborsClassifier(n_neighbors = n)
    model.fit(dataset.data, dataset.target)
    #print(model)
    # make predictions
    #expected = dataset.target
    #predicted = model.predict(dataset.data)
    # summarize the fit of the model
    ###mse = np.mean((predicted-expected)**2)
    #print(mse)
    print(model.score(dataset.data, dataset.target))

print ('distance')

for j in range(1, 21):
    model = KNeighborsClassifier(weights = 'distance', n_neighbors = j)
    model.fit(dataset.data, dataset.target)
    ###print(model)
    # make predictions
    #expected = dataset.target
    #predicted = model.predict(dataset.data)
    # summarize the fit of the model
    ###mse = np.mean((predicted-expected)**2)
    #print(mse)
    print(model.score(dataset.data, dataset.target))



