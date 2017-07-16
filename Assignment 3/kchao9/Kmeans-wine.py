import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn import mixture

import pandas as pd


centers = [[1, 1], [-1, -1], [1, -1]]
#iris = datasets.load_iris()
#X = iris.data
###y = iris.target


mydata = pd.read_csv('wine.csv', header = None)
dataset = mydata
dataset.target = mydata.ix[:, 13]
dataset.data = mydata.ix[:,0:12]
X = dataset.data
y = dataset.target

print (dataset.target)

estimators = {'k_means_iris_3': KMeans(n_clusters=3),
              'k_means_iris_8': KMeans(n_clusters=8),
              'k_means_iris_bad_init': KMeans(n_clusters=3, n_init=1,init='random'),
              'Expectation Maximization' :mixture.GaussianMixture(n_components=5, covariance_type='full')}


fignum = 1
for name, est in estimators.items():
    fig = plt.figure(fignum, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

    plt.cla()
    est.fit(X)
    labels = est.labels_

    ax.scatter(X.ix[:, 6], X.ix[:, 10], X.ix[:, 11], c=labels.astype(np.float))

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('Flavanoids ')
    ax.set_ylabel('Hue')
    ax.set_zlabel('OD280/OD315')
    fignum = fignum + 1

# Plot the ground truth
fig = plt.figure(fignum, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()

for name, label in [('Iris-setosa', 0),
                   ('Iris-versicolor', 1),
                   ('Iris-virginica', 2)]:
    ax.text3D(X[y == label, 3].mean(),
              X[y == label, 0].mean() + 1.5
           X[y == label, 2].mean(), name,
              horizontalalignment='center',
              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
 Reorder the labels to have colors matching the cluster results

y = np.choose(y, [1, 2, 0]).astype(np.float)
ax.scatter(X.ix[:, 3], X.ix[:, 0], X.ix[:, 2], c=y)

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Flavanoids')
ax.set_ylabel('Hue')
ax.set_zlabel('OD280/OD315')
plt.show()


