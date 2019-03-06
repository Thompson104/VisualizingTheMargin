# Loss is quadratic so we just have to fit residuals. 
from sklearn.tree import DecisionTreeClassifier # default weak learner
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import matplotlib.pyplot as plt
import copy
from sklearn.linear_model import *

class GradientBoosting:

	def __init__(self, T=100, base_learner=DecisionTreeRegressor(max_depth=1)):
		self.base_learner = base_learner
		self.T = T
		self.estimators = []
		self.residuals 	= None
		self.errors 	= []
		self.test_errors 		= []



	def step(self, X, y): 

		if self.residuals is None: 
			self.residuals 	= np.copy(y) # Let initial model nothing, so residual is y. 
			self.pred 		= np.zeros(y.shape)


		h_t = copy.deepcopy(self.base_learner)
		h_t.fit(X, self.residuals)

		# add model to ensemble
		self.estimators.append(h_t)

		# update residuals
		self.pred 		+= h_t.predict(X)
		#pred_ 		= self.predict(X)
		self.residuals 	= y - self.pred
		#assert np.allclose(pred, pred_)

		
	def normalized_margins(self, X, y): return self.predict(X) * y / len(self.estimators)

	def predict(self, X):
		n, d = X.shape
		predictions = np.sum([est.predict(X) for est in self.estimators], axis=0)
		return predictions

	def score(self, X, y): return np.mean(np.sign(self.predict(X)) == y)

	def error(self, X, y): return 1 - self.score(X, y)

	def cost(self, X, y): return np.sum( (self.predict(X) - y)**2)



