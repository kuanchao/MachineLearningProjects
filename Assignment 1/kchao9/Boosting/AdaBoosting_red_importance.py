import numpy as np
from sklearn import datasets
from sklearn import ensemble

from sklearn.model_selection import learning_curve


import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier


# load the datasets
#dataset = datasets.load_diabetes()
# fit a model to the data
import pandas as pd
mydata = pd.read_csv('winequality-red.csv')
dataset = mydata
dataset.target = mydata["quality"]
#provided your csv has header row, and the label column is named "Label"
dataset.feature_names = np.array(list(mydata))
#select all but the last column as data
dataset.data = mydata.ix[:,:-1]
model = ensemble.AdaBoostClassifier()
model.fit(dataset.data, dataset.target)



importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(dataset.data.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(dataset.data.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(dataset.data.shape[1]), dataset.feature_names[indices])
plt.xlim([-1, dataset.data.shape[1]])
plt.show()








