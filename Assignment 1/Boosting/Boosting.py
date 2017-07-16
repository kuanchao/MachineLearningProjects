import numpy as np
import matplotlib.pyplot as plt

from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction import DictVectorizer

from sklearn.model_selection import learning_curve

import pandas as pd
mydata = pd.read_csv('winequality-red.csv')
dataset = mydata
dataset.target = mydata["quality"]

###boston = datasets.load_boston()

dataset.data = mydata.ix[:,:-1]



X, y = shuffle(dataset.data, dataset.target, random_state=11)
X = X.astype(np.float32)
offset = int(X.shape[0] * 0.9)
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)

clf.fit(X_train, y_train)
mse = mean_squared_error(y_test, clf.predict(X_test))
score = clf.score(X_test, y_test)
print(score)
print("MSE: %.4f" % mse)

train_sizes, train_scores, valid_scores = learning_curve(clf, dataset.data,
dataset.target, train_sizes=[100, 200, 300, 400, 500, 600, 700, 800], cv=5)
train_sizes
print (train_scores)
print (valid_scores)

"""# compute test set deviance
test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

for i, y_pred in enumerate(clf.staged_predict(X_test)):
    test_score[i] = clf.loss_(y_test, y_pred)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',
         label='Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
         label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')



feature_importance = clf.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos)
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()"""
