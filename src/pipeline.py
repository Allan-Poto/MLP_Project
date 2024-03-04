# General
from sklearn.pipeline import Pipeline

# Preprocessing
from feature_engine import encoding as ce
from feature_engine import imputation as mdi
from feature_engine import selection as sel

# Models
from sklearn.linear_model import LogisticRegression # Baseline Model
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier

from config.core import config
from processing import feateng

if config.MODEL_SELECTION not in config.MODEL_LIST:
	raise ValueError(f'Available models are: {config.MODEL_LIST}. To continue, add your model to MODEL_LIST in pipeline.py')

## PREPROCESSING PIPELINE
pp_pipeline = Pipeline(
	[
		(
			"basic_cleaning",
			feateng.BasicCleaning()
		),
		(
			"impute_median",
			mdi.MeanMedianImputer(
				variables=config.IMPUTE_MEDIAN
			)
		),
		(
			"impute_mode",
			mdi.CategoricalImputer(
				imputation_method='frequent',
				variables=config.IMPUTE_MODE,
				return_object=True,
				ignore_format=True
			)
		),
		(
			"impute_reference_weight",
			feateng.ReferenceImputer(
				channel="weight",
				var1=config.IMPUTE_REFERENCE[0][0],
				var2=config.IMPUTE_REFERENCE[0][1]
			)
		),
		(
			"impute_reference_medical",
			feateng.ReferenceImputer(
				channel="medical",
				var1=config.IMPUTE_REFERENCE[1][0], 
				var2=config.IMPUTE_REFERENCE[1][1]
			)
		),
		(
			"create_feature",
			feateng.FeatureCreator()
		),
		(
			"drop",
			feateng.FeatureDropper(
				drop_vars=config.DROP_FEATURES
			)

		)
	]
)

## ENCODING PIPELINES
ec_pipeline = Pipeline(
	[
		("nominal_encode",
			ce.OneHotEncoder(
				variables=config.NOMINAL_CATEGORICAL
			)
		),
		("ordinal_encode",
			ce.OrdinalEncoder(
				variables=config.ORDINAL_CATEGORICAL
			)
		)
	])

## FEATURE SELECTION PIPELINE
fs_pipeline = Pipeline(
	[
		(
			"duplicates", 
			sel.DropDuplicateFeatures()
		),
		(
			"corr",
			sel.SmartCorrelatedSelection(
				selection_method="variance"
			)
		)
	]
)

## ERROR CHECKING PIPELINE
#When you want to check how the dataframe looks like after the processing pipeline
check_pipeline = Pipeline(
	[
		(
			"check",
			feateng.PipelineChecker()
		)
	])

## PREPROCESSING COMBINATION PIPELINES
# processing_pipeline = Pipeline(
# 	[("pp", pp_pipeline), ("ec", ec_pipeline), ("fs", fs_pipeline), ("cp", check_pipeline)]
# )

processing_pipeline = Pipeline(
	[("pp", pp_pipeline), ("ec", ec_pipeline), ("fs", fs_pipeline)]
)



## MODEL PIPELINES
# TODO: ADD tuning_final_pipeline WHERE a model is trained using the OPTIMIZED HYPERPARAMTERS AFTER OPTIMIZATION TUNING
if config.MODEL_SELECTION == "LogisticRegression":
	tuning_pipeline = Pipeline(
		[
			("total", processing_pipeline),
			("model",
    				LogisticRegression(
					verbose=1,
					random_state=config.SEED
				)
			)
		])
	model_pipeline = Pipeline(
		[
			("total", processing_pipeline),
			("model",
				LogisticRegression(
					verbose=1,
					**config.LOG_STR_HPARAMS,
					**config.LOG_INT_HPARAMS,
					**config.LOG_FLOAT_HPARAMS,
					random_state=config.SEED
				)
			)
		])
elif config.MODEL_SELECTION == "LGBMClassifier":
	tuning_pipeline = Pipeline(
		[
			("total", processing_pipeline),
			("model",
				LGBMClassifier(
					objective="binary",
					boosting_type="gbdt",
					random_state=config.SEED
				)
			)
		])

	model_pipeline = Pipeline(
		[
			("total", processing_pipeline),
			("model",
				LGBMClassifier(
					objective="binary",
					boosting_type="gbdt",
					**config.LGBM_STR_HPARAMS,
					**config.LGBM_INT_HPARAMS,
					**config.LGBM_FLOAT_HPARAMS,
					random_state=config.SEED
				),
			)
		])
elif config.MODEL_SELECTION == "SVC":
	tuning_pipeline = Pipeline(
		[
			("total", processing_pipeline),
			("model",
				SVC(
					gamma="auto",
					verbose=True
				)
			)
		])
	model_pipeline = Pipeline(
		[
			("total", processing_pipeline),
			("model",
				SVC(
					gamma="auto",
					**config.SVC_STR_HPARAMS,
					**config.SVC_FLOAT_HPARAMS,
					verbose=True
				)
			)
		])
elif config.MODEL_SELECTION == "MLPClassifier":
	tuning_pipeline = Pipeline(
		[
			("total", processing_pipeline),
			("model",
				MLPClassifier(
					max_iter=500,
					verbose=True,
					random_state=config.SEED,
					early_stopping=True
				)
			)
		])
	model_pipeline = Pipeline(
		[
			("total", processing_pipeline),
			("model",
				MLPClassifier(
					**config.MLP_STR_HPARAMS,
					**config.MLP_INT_HPARAMS,
					**config.MLP_FLOAT_HPARAMS,
					max_iter=500,
					verbose=True,
					random_state=config.SEED,
					early_stopping=True
				)
			)
		])
else:
	raise ValueError(f'Available models are: {config.MODEL_LIST}. To continue, add your model to MODEL_LIST in pipeline.py')