import numpy as np
import pandas as pd
import datetime as dt
from typing import List

from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import StandardScaler

CURR_DATE = dt.datetime.today()

class BasicCleaning(BaseEstimator, TransformerMixin):
	"""
	Conduct Basic Cleaning including the following:
	1. Fix incorrect input like negative values
	2. Fix possible string formatting issues in all columns
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
		X["Gender"] = [gender.lower().strip() if gender.upper().strip() != "NAN" else None for gender in X["Gender"]]
		X["COPD History"] = [copd.lower().strip() if copd is not None else copd for copd in X["COPD History"]]
		X["Taken Bronchodilators"] = [tb.lower().strip() if tb is not None else tb for tb in X["Taken Bronchodilators"]]
		X["Genetic Markers"] = [gm.lower().strip() if gm is not None else gm for gm in X["Genetic Markers"]]
		X["Air Pollution Exposure"] = [ape.lower().strip() if ape is not None else ape for ape in X["Air Pollution Exposure"]]
		X["Frequency of Tiredness"] = [fot.lower().strip() if fot is not None else fot for fot in X["Frequency of Tiredness"]]
		X["Dominant Hand"] = [dh.lower().strip() if dh is not None else dh for dh in X["Dominant Hand"]]
		
		## Other Variables
		X["Start Smoking"] = [sts.lower().strip() if sts is not None else sts for sts in X["Start Smoking"]]
		X["Stop Smoking"] = [sps.lower().strip() if sps is not None else sps for sps in X["Stop Smoking"]]
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
			# IF EITHER VALUE IS PRESENT, COPY OVER THE VALUE TO THE OTHER ONE
			if X.loc[i, self.var1] is None:
				if X.loc[i, self.var2] is not None:
					X.loc[i, self.var1] = X.loc[i, self.var2]
			if X.loc[i, self.var2] is None:
				if X.loc[i, self.var1] is not None:
					X.loc[i, self.var2] = X.loc[i, self.var1]
			
			# IF BOTH VALUES ARE NOT PRESENT, 
			if X.loc[i, self.var1] is None and X.loc[i, self.var2] is None:
				if self.channel == "weight": # FOR WEIGHT, IMPUTE USING MEDIAN
					X.loc[i, self.var1] = X[self.var1].median()
					X.loc[i, self.var2] = X[self.var2].median()
				else: # CREATE AN ADDITIONAL LABEL CALLED UNKNOWN FOR MEDICAL
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
		return X

class FeatureDropper(BaseEstimator, TransformerMixin):
	"""
	Drop the columns as stated in configurations
	"""
	def __init__(self, drop_vars: List[str]):
		self.drop_vars = drop_vars

	def fit(self, X: pd.DataFrame, y: pd.Series = None):
		return self

	def transform(self, X: pd.DataFrame):
		X = X.copy()
		for feature in self.drop_vars:
			if feature in X.columns:
				X = X.drop(feature,axis=1)
		return X

class PipelineChecker(BaseEstimator, TransformerMixin):
	"""
	Insert when you want to check on the DataFrame at any point in the Pipeline
	"""
	def fit(self, X: pd.DataFrame, y: pd.Series = None):
		return self
	
	def transform(self, X: pd.DataFrame):
		X = X.copy()
		print(X.head())
		print(X.info())
		
		return X

## Not working properly
# class NumericalScaler(BaseEstimator, TransformerMixin):
# 	"""
# 	Apply standardscaler and transforms the output back into a dataframe 
# 	"""
# 	def __init__(self, variables: List[str]):
# 		self.num_vars = variables

# 	def fit(self, X: pd.DataFrame, y: pd.Series = None):
# 		return self

# 	def transform(self, X: pd.DataFrame):
# 		X = X.copy()
# 		colnames = X.columns.tolist()
# 		for col in self.num_vars:
# 			colnames.remove(col)
# 		colnames = self.num_vars + colnames
# 		ct = ColumnTransformer([('scaler', StandardScaler(), self.num_vars)],remainder='passthrough')
# 		X_TRANSFORM = ct.fit_transform(X)
# 		X = pd.DataFrame(X_TRANSFORM, columns=colnames)
# 		print(X.info())
# 		return X
