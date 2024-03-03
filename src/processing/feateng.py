from typing import List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ReferenceImputer(BaseEstimator, TransformerMixin):
	"""
	Imputing a column that have some relationship with 
	another column

	Consists of 2 channels (switch using method set_channel):
	 - weight
	 - medical

	channel weight:
		Inputs (Order does not matter):
		 - "Current Weight"
		 - "Last Weight"
	channel medical:
		Inputs (Order does not matter):
		 - "COPD History"
		 - "Taken Bronchodilators"
	"""
	global channels
	channels = ["weight", "medical"]

	def __init__(self, var1: str, var2: str):
		if (not isinstance(var1, str)) or (not isinstance(var2, str)):
			raise ValueError("Inputs should be a string")

		self.channel = None
		self.var1 = var1
		self.var2 = var2

	def set_channel(self, channel):
		if channel not in channels:
			raise ValueError(f'Valid Channels are {channels}')
		self.channel = channel

	def fit(self, X: pd.DataFrame, y: pd.Series = None):
		return self

	def transform(self, X: pd.DataFrame):
		X = X.copy()
		X = X.replace(np.nan, None)
		col1= X[self.var1].copy()
		col2= X[self.var2].copy()
		
		for i in range(len(col1)):
			if X.loc[i, self.var1] is None:
				if X.loc[i, self.var2] is not None:
					X.loc[i, self.var1] = X.loc[i, self.var2]
			if X.loc[i, self.var2] is None:
				if X.loc[i, self.var1] is not None:
					X.loc[i, self.var2] = X.loc[i, self.var1]
			if X.loc[i, self.var1] is None and X.loc[i, self.var2] is None:
				if self.channel == "weight":
					X.loc[i, self.var1] = X[self.var1].median()
					X.loc[i, self.var2] = X[self.var2].median()
				else:
					X.loc[i, self.var1]  = "Unknown"
					X.loc[i, self.var2]  = "Unknown"
		return X


class FeatureCreator(BaseEstimator, TransformerMixin):
	"""
	Creates the engineered features required and drop the
	columns used to prevent multicollinearity
	"""

	def fit(self, X: pd.DataFrame, y: pd.Series = None):
		return self

	def transform(self, X: pd.DataFrame):
		X = X.copy()

		# NUMERICAL
		X["Weight Change"] = X["Current Weight"] - X["Last Weight"]
		X = X.drop(["Current Weight", "Last Weight"], axis=1)
		
		return X