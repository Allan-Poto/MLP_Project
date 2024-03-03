import numpy as np
import pandas as pd
import datetime as dt
from typing import List
from sklearn.base import BaseEstimator, TransformerMixin

CURR_DATE = dt.datetime.today()

class BasicCleaning(BaseEstimator, TransformerMixin):
	"""
	Conduct Basic Cleaning including the following:
	1. Drop duplicated rows
	2. Fix incorrect input like negative values
	3. Fix string format issues
	"""

	def fit(self, X: pd.DataFrame, y: pd.Series=None):
		return self
	
	def transform(self, X: pd.DataFrame):
		X = X.copy()

		# Numerical Variables
		X["Age"] = X["Age"].abs()
		X["Current Weight"] = X["Current Weight"].abs()
		X["Last Weight"] = X["Last Weight"].abs()

		# Categorical Variables
		X["Gender"] = [gender.lower().strip() if gender != "NAN" else None for gender in X["Gender"]]
		X["COPD History"] = [copd.lower().strip() if copd is not None else copd for copd in X["COPD History"]]
		X["Taken Bronchodilators"] = [tb.lower().strip() if tb is not None else tb for tb in X["Taken Bronchodilators"]]
		X["Genetic Markers"] = [gm.lower().strip() if gm is not None else gm for gm in X["Genetic Markers"]]
		X["Air Pollution Exposure"] = [ape.lower().strip() if ape is not None else ape for ape in X["Air Pollution Exposure"]]
		X["Frequency of Tiredness"] = [fot.lower().strip() if fot is not None else fot for fot in X["Frequency of Tiredness"]]
		X["Dominant Hand"] = [dh.lower().strip() if dh is not None else dh for dh in X["Dominant Hand"]]
		
		## Other Variables
		X["Start Smoking"] = [sts.lower().strip() if sts is not None else sts for sts in X["Start Smoking"]]
		X["Stop Smoking"] = [sps.lower().strip() if sps is not None else sps for sps in X["Stop Smoking"]]
		print(f'After Cleaning: {len(X)}')
		return X


class ReferenceImputer(BaseEstimator, TransformerMixin):
	"""
	Imputing a column that have some relationship with 
	another column

	Consists of 2 channels (switch using method set_channel):
	 - weight
	 - medical

	channel weight:
	- Inputs (Order does not matter):
		- "Current Weight"
		- "Last Weight"
	
	channel medical:
	- Inputs (Order does not matter):
		- "COPD History"
		- "Taken Bronchodilators"
	"""
	global channels
	channels = ["weight", "medical"]

	def __init__(self, channel: str, var1: str, var2: str):
		if (not isinstance(var1, str)) or (not isinstance(var2, str)):
			raise ValueError("Inputs should be a string")

		self.channel = channel
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
		
		for i in X.index:
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

		# WEIGHT
		X["Weight Change"] = X["Current Weight"] - X["Last Weight"]
		
		# SMOKING
		mapping = {"not applicable": 0, "still smoking": CURR_DATE.year}
		years_smoked = []
		for start, stop in zip(X["Start Smoking"], X["Stop Smoking"]):
			if len(start) != 4:
				start = mapping[start]
			if len(stop) != 4:
				stop = mapping[stop]
			years_smoked.append(int(stop)-int(start))
		X["Years Smoked"] = years_smoked

		CAT_SMOKER = []

		for years in X["Years Smoked"]:
			if years <= 5:
				CAT_SMOKER.append("Short Term")
			elif years > 5 and years <= 20:
				CAT_SMOKER.append("Middle Term")
			else:
				CAT_SMOKER.append("Long Term")

		X["Cat Smoker"] = CAT_SMOKER
		
		print(f'After Feature Creation: {len(X)}')
		return X
	
class FeatureDropper(BaseEstimator, TransformerMixin):
	"""
	Creates the engineered features required and drop the
	columns used to prevent multicollinearity
	"""
	def __init__(self, drop_vars: List[str]):
		self.drop_vars = drop_vars

	def fit(self, X: pd.DataFrame, y: pd.Series = None):
		return self

	def transform(self, X: pd.DataFrame):
		X = X.copy()
		X = X.drop(self.drop_vars,axis=1)
		print(f'After feature dropping: {len(X)}')

		return X