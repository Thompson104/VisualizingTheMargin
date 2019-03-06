"""
	Inspired by https://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_twoclass.html

"""

import numpy as np
import matplotlib.pyplot as plt

from AdaBoost import AdaBoost
from GradientBoosting import GradientBoosting
from Bagging import Bagging

from sklearn.tree import DecisionTreeClassifier

from sklearn.datasets import make_gaussian_quantiles


# 
fig, ax = plt.subplots(3, 3, figsize=(9, 9)) # one for each classifier ;; add more rows, one row for margin histogram, one for error and one for decision boundary. 
fig.canvas.manager.window.wm_geometry("+2000+0")

def plot_step(vote, col, X, y, title=None): 
	vote.errors.append(1 - vote.score(X_train, y_train))
	vote.test_errors.append(1 - vote.score(X_val, y_val))
	margins = vote.normalized_margins(X_train, y_train) 
	margins = np.sort(margins)

	ax[0,col].cla()
	ax[0,col].plot([0, len(vote.errors)], [vote.errors[0], vote.errors[0]], label="Base Train Error", alpha=0.3)
	ax[0,col].plot([0, len(vote.errors)], [vote.test_errors[0], vote.test_errors[0]], label="Base Val Error", alpha=0.3)
	ax[0,col].plot(np.arange(len(vote.errors)), vote.errors, label="Voting Train Error")
	ax[0,col].plot(np.arange(len(vote.test_errors)), vote.test_errors, label="Voting Val Error")
	ax[0,col].set_ylim([0, 0.6])
	ax[0,col].set_ylabel("Error (%)")
	ax[0,col].set_xlabel("# base classifiers")
	if col == 2: ax[0,col].legend(loc=1)

	ax[1,col].cla()
	ax[1,col].plot(margins, np.arange(margins.size) / margins.size)
	ax[1,col].set_ylabel("Cumulative Distribution")
	ax[1,col].set_xlabel("Margin")
	ax[1,col].set_ylim([0, 1])
	ax[1,col].set_xlim([-1, 1])

	ax[2, col].cla()
	plot_step = 0.02
	x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))

	Z = np.sign(vote.predict( np.c_[xx.ravel(), yy.ravel()] ) )
	Z = Z.reshape(xx.shape)
	ax[2,col].contourf(xx, yy, Z, cmap=plt.cm.Paired)
	
	ax[2,col].plot(X_train[y_train==-1][:, 0], X_train[y_train==-1][:, 1], 'bx', alpha=0.3)
	ax[2,col].plot(X_train[y_train==1][:, 0], X_train[y_train==1][:, 1], 'r^', alpha=0.3)

	ax[2, col].axis('off')


	if not title is None: ax[0, col].set_title(title)

	plt.pause(.1)



# Construct dataset ;; not really hard enough... anyway, also plot decision boundary 
X1, y1 = make_gaussian_quantiles(cov=2.,
                                 n_samples=400, n_features=2,
                                 n_classes=2, random_state=1)
X2, y2 = make_gaussian_quantiles(mean=(3, 3), cov=1.5,
                                 n_samples=400, n_features=2,
                                 n_classes=2, random_state=1)
X = np.concatenate((X1, X2))
y = np.concatenate((y1, - y2 + 1)) * 2 - 1

n, d = X.shape

perm = np.random.permutation(n)
X = X[perm]
y = y[perm]

X_train = X[:n//2]
y_train = y[:n//2]

X_val 	= X[n//2:]
y_val 	= y[n//2:]

def run(max_depth=1): 

	ada = AdaBoost(			base_learner=DecisionTreeClassifier(max_depth=max_depth))
	gb 	= GradientBoosting(	base_learner=DecisionTreeClassifier(max_depth=max_depth))
	bag = Bagging(			base_learner=DecisionTreeClassifier(max_depth=max_depth))

	fig.suptitle("Max Base Learner Tree Depth: %i"%max_depth)

	for i in range(50): 

		print("--- [%i / %i] ---"%(i+1, 50))

		ada.step(X_train, y_train)
		gb.step(X_train, y_train)
		bag.step(X_train, y_train)

		plot_step(ada, 	0, X, y, title="AdaBoost")
		plot_step(bag, 	1, X, y, title="Bagging")
		plot_step(gb, 	2, X, y, title="Gradient Boosting")

		if i == 0: plt.tight_layout(rect=[0, 0, 1, 0.95])

		plt.savefig("images/depth_%i/%i.jpg"%(max_depth, i))


run(1)
run(2)
run(4)
run(8)
