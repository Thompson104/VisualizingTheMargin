from sklearn.tree import DecisionTreeClassifier # default weak learner
import numpy as np
import matplotlib.pyplot as plt
import copy

class AdaBoost:


	def __init__(self, T=100, base_learner=DecisionTreeClassifier(max_depth=1)):
		self.alphas		  		= []
		self.estimators	  		= []
		self.sample_weight 		= None
		self.base_learner 		= base_learner
		self.T 					= T

		self.errors 			= []
		self.test_errors 		= []


	def step(self, X, y, plot=None):
		""" Learn a single hypothesis and add it to ensemble. """
		n, d = X.shape	

		# If sample weights is None set uniform sample weight.
		if self.sample_weight is None: self.sample_weight = np.ones(n) / n
		
		# Make sure that sample weight is distribution, i.e. has positive entries that sum to 1.
		assert np.allclose(np.sum(self.sample_weight), 1)
		assert np.sum(self.sample_weight < 0) == 0

		# Learn a weak classifier on weighted data
		h_t  = copy.deepcopy(self.base_learner)  
		h_t.fit(X, y, sample_weight=self.sample_weight)
		
		# Compute error of current hypothesis
		eps_t = 1 - h_t.score(X, y, sample_weight=self.sample_weight)

		# If perfect we are done. 
		assert eps_t != 0

		# Compute weight of current hypothesis given its error
		alpha_t = 1/2 * np.log((1-eps_t)/eps_t)

		# Add weight and weak classifier to lists
		self.estimators.append(h_t)
		self.alphas.append(alpha_t)

		# Update sample weight to favour points we perform bad on.
		predictions 		= h_t.predict(X)
		self.sample_weight 	= self.sample_weight * np.exp(- alpha_t * y * predictions ) 
		h_t.Z 				= np.sum(self.sample_weight)
		h_t.alpha_t 		= alpha_t
		self.sample_weight 	= self.sample_weight / h_t.Z


	def predict_unsigned(self, X):
		n, d = X.shape

		pred = [self.alphas[i]*self.estimators[i].predict(X) for i in range(len(self.estimators))]
		pred = np.sum(pred, axis=0)

		return pred

	def predict(self, X): 			return np.sign(self.predict_unsigned(X))

	def margins(self, X, y): 		return self.predict_unsigned(X) * y

	def normalized_margins(self, X, y): return self.margins(X, y)  / np.sum(self.alphas)
	
	def score(self, X, y): 			return np.mean(self.predict(X) == y) 
	
	def error(self, X, y): 			return 1 - self.score(X, y)

