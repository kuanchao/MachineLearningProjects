# -*- coding: cp1252 -*-
# Code source: Gaël Varoquaux
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.random_projection import GaussianRandomProjection
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import learning_curve

from sklearn.cluster import KMeans
import numpy as np

mydata = pd.read_csv('wine.csv', header = None)
dataset = mydata
dataset.target = mydata.ix[:, 13]
dataset.data = mydata.ix[:,0:12]
X = dataset.data
Y = dataset.target



plt.figure(2, figsize=(8, 6))
plt.clf()

# Plot the training points
"""plt.scatter(X.ix[:,6], X.ix[:,10], c=Y, cmap=plt.cm.Paired)
plt.xlabel('Flavanoids')
plt.ylabel('Hue')

x_min, x_max = X.ix[:,6].min() - .5, X.ix[:,6].max() + .5
y_min, y_max = X.ix[:,10].min() - .5, X.ix[:,10].max() + .5

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
"""
# To getter a better understanding of interaction of the dimensions
# plot the first three Random Projection dimensions
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = GaussianRandomProjection(n_components=3).fit_transform(X)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=Y)
ax.set_title("First three Random Projection directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])

fig = plt.figure(3, figsize=(8, 6))

X_reduced = GaussianRandomProjection(n_components=3).fit_transform(X)

plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=Y)

plt.xlabel('1st eigenvector')
plt.ylabel('2nd eigenvector')

estimators = {'k_means_iris_3': KMeans(n_clusters=3),
              'k_means_iris_8': KMeans(n_clusters=8),
              'k_means_iris_bad_init': KMeans(n_clusters=3, n_init=1,
                                              init='random')}


fignum = 5
for name, est in estimators.items():
    fig = plt.figure(fignum, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

    plt.cla()
    est.fit(X_reduced)
    labels = est.labels_

    ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=labels.astype(np.float))

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('1st eigenvector')
    ax.set_ylabel('2nd eigenvector')
    ax.set_zlabel('3rd eigenvectorh')
    fignum = fignum + 1

# Plot the ground truth
fig = plt.figure(fignum, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()


ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('1st eigenvectorh')
ax.set_ylabel('2nd eigenvector')
ax.set_zlabel('3rd eigenvector')


model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
model.fit(X_reduced, Y)
print(model)
print(model.score(X_reduced, Y))

train_sizes, train_scores, valid_scores = learning_curve(model, X_reduced,Y, train_sizes=[ 35,50,65,80, 95], cv=5)

print (train_scores)
print (valid_scores)


plt.show()
