from feature_engine import encoding as ce
from feature_engine import imputation as mdi
from feature_engine import selection as sel
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from src.config.core import config
from src.processing import feateng

pp_pipeline = Pipeline(
	[
		(
			"basic_cleaning",
			feateng.BasicCleaning()
		),
		(
			"impute_median",
			mdi.MeanMedianImputer(
				variables=config["IMPUTE_MEDIAN"]
			)
		),
		(
			"impute_mode",
			mdi.CategoricalImputer(
				fill_value="Missing",
				variables=config["IMPUTE_MODE"],
				return_object=True,
				ignore_format=True
			)
		),
		(
			"impute_reference_weight",
			feateng.ReferenceImputer(
				channel="weight",
				var1=config["IMPUTE_REFERENCE"][0][0],
				var2=config["IMPUTE_REFERENCE"][0][1]
			)
		),
		(
			"impute_reference_medical",
			feateng.ReferenceImputer(
				channel="medical",
				var1=config["IMPUTE_REFERENCE"][1][0], 
				var2=config["IMPUTE_REFERENCE"][1][1]
			)
		),
		(
			"create_feature",
			feateng.FeatureCreator()
		)
	]
)

ec_pipeline = Pipeline(
    [
	(
	    "cat_encoder",
	    ce.OneHotEncoder(
			variables=config["ENCODING_CATEGORICAL"],
			ignore_format=True
	    )
	)
    ]
)

fs_pipeline = Pipeline(
    [
	(
		"drop", 
		sel.DropFeatures(config["DROP_FEATURES"])
	),
	(
		"constant", 
		sel.DropConstantFeatures()
	),
	(
		"duplicates", 
		sel.DropDuplicateFeatures()
	),
	(
	    "correl",
	    sel.SmartCorrelatedSelection(
		selection_method="model_performance",
		estimator=DecisionTreeRegressor(
		    random_state=config["SEED"]
		),
		scoring="neg_mean_squared_error",
	    ),
	),
    ]
)

total_pipeline = Pipeline(
    [("pp", pp_pipeline), ("ec", ec_pipeline), ("fs", fs_pipeline)]
)
