# -*- coding: cp1252 -*-
# Code source: Gaël Varoquaux
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets

import pandas as pd
from sklearn.decomposition import FactorAnalysis

import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import learning_curve



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
# plot the first three FactorAnalysis dimensions
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = FactorAnalysis(n_components=3).fit_transform(X)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=Y)
ax.set_title("First three FactorAnalysis directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])

import itertools

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture

from sklearn import datasets

import pandas as pd

color_iter = itertools.cycle(['navy','red', 'c', 'cornflowerblue', 'gold',
                              'darkorange'])


def plot_results(X, Y_, means, covariances, index, title):
    splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X_reduced[Y_ == i,0], X_reduced[Y_ == i,1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)


    plt.xticks(())
    plt.yticks(())
    plt.title(title)


# Fit a Gaussian mixture with EM using three components
gmm = mixture.GaussianMixture(n_components=3, covariance_type='full').fit(X_reduced)
plot_results(X_reduced, gmm.predict(X_reduced), gmm.means_, gmm.covariances_, 0,
             'Gaussian Mixture')

# Fit a Dirichlet process Gaussian mixture using three components
dpgmm = mixture.BayesianGaussianMixture(n_components=3,
                                        covariance_type='full').fit(X_reduced)
plot_results(X_reduced, dpgmm.predict(X_reduced), dpgmm.means_, dpgmm.covariances_, 1,
             'Bayesian Gaussian Mixture with a Dirichlet process prior')

model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
model.fit(X_reduced, Y)
print(model)
print(model.score(X_reduced, Y))

train_sizes, train_scores, valid_scores = learning_curve(model, X_reduced,
    dataset.target, train_sizes=[ 35,50,65,80, 95], cv=5)
print (train_scores)
print (valid_scores)


plt.show()
