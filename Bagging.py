from sklearn.tree import DecisionTreeClassifier # default weak learner
import numpy as np
import matplotlib.pyplot as plt
import copy

class Bagging:

	def __init__(self, T=100, base_learner=DecisionTreeClassifier(max_depth=1)):
		self.estimators	  		= []
		self.base_learner 		= base_learner
		self.T 					= T
		self.errors 			= []
		self.test_errors 		= []

		
	def step(self, X, y, plot=None):
		""" Learn a single hypothesis and add it to ensemble. """
		n, d = X.shape
		
		indxs = np.random.choice(np.arange(n), n//2, replace=True)
		bootstrap_sample_X = X[indxs]
		bootstrap_sample_y = y[indxs]

		# Learn a weak classifier on bootstrap sample. 
		h_t  = copy.deepcopy(self.base_learner)  
		h_t.fit(bootstrap_sample_X, bootstrap_sample_y)
		
		# Add weak classifier to list
		self.estimators.append(h_t)

	def predict_unsigned(self, X):
		n, d = X.shape

		pred = [estimator.predict(X) for estimator in self.estimators]
		pred = np.sum(pred, axis=0)

		return pred

	def predict(self, X): 			return np.sign(self.predict_unsigned(X))

	def margins(self, X, y): 		return self.predict_unsigned(X) * y

	def score(self, X, y): 			return np.mean(self.predict(X) == y) 
	
	def error(self, X, y): 			return 1 - self.score(X, y)

	def hypothesis_count(self): 	return len(self.estimators)

	def normalized_margins(self, X, y): return self.margins(X, y) / len(self.estimators) # each guy has weight 1. 


