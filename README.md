# README for AISG Submission

## **a. Full name (as in NRIC) and email address.**

Full name: Tay Wei Hong, Allan

Email address: <whtay.allan@gmail.com>

## **b. Overview of the submitted folder and the folder structure.**

```
root
├───.github
│   └───workflows
│───src
│   ├───config
│   │   └───__init__.py
│   │   └───core.py 
│   │───data
│   │   └───lung_cancer.db
│   │───model
│   │   └───*__init__.py
│   │   └───*.pkl (Trained Models)
│   │───processing
│   │   └───__init__.py
│   │   └───feateng.py
│   │───config.yml
│   │───main.py 
│   │───pipeline.py
│   │───predict.py
│   │───train.py (Contains Training functions)
│   └───utils.py (Contains Reuseable functions)
│───.gitignore
│───eda.ipynb
│───README.md
│───requirements.txt
└───run.sh
```

Main Files Breakdown:

- [core.py](src/config/core.py): Config File Management and Validation
- [feateng.py](src/processing/feateng.py): Custom Transformers for cleaning, feature_creation for Pipeline
- [config.yml](src/config.yml): Data and Model Configurations
- [main.py](src/main.py): Main program to call upon to execute the full MLP
- [pipeline.py](src/pipeline.py): Contains all Pipelines to be used in creating the model
- [predict.py](src/predict.py): Load model and make predictions
- [train.py](src/train.py): Train and Optimize model
- [utils.py](src/utils.py): Reusable functions for data and model management
- [eda.ipynb](eda.ipynb): Python notebook with EDA and their notes
- [requirements.txt](requirements.txt): Dependencies to be downloaded
- [run.sh](run.sh): Shell script for running the main.py program

## **c. Instructions for executing the pipeline and modifying any parameters.**

### Execution of the pipeline

1. From the root folder, run `pip3 install -r requirements.txt`. This should load up all necessary dependencies for the EDA and for the main program

2. Download the dataset from  <https://techassessment.blob.core.windows.net/aiap16-assessment-data/lung_cancer.db> and place it in the [data folder](src/data/).

3. Run `./run.sh`. Upon execution of the shell script, it will trigger an execution of the [main.py](src/main.py), which will proceed to train a model on the downloaded dataset using the current configurations set in [config.yml](src/config.yml). The trained model will be stored in the [model folder](src/model/). The program will then proceed to load the model, make predictions on the test data and print out evaluation (recall and F1_Score) for both the training and testing set.

TODO: API Implementation of training/predicting where the model trains on the whole dataset and have an API point to take a file as input and make predictions on that file.

### Modifications

#### MODELS

Currently within the pipeline, there are 4 (including the baseline) models available:

1. Baseline Model `LogisticRegression` > sklearn
2. `Support Vector Classifier` > sklearn
3. `MLPClassifier` > sklearn
4. `Light Gradient Boosting Machine Classifier` > lightgbm

To add on more models, you will have to proceed with the following steps:

1. pip install the relevant libraries for those models if they are not from the `scikit-learn` or `lightgbm` libraries.
2. Navigate to [config.yml](src/config.yml) and add your model_name to the parameter `MODEL_LIST` in the MODELLING segment. Change the parameter `MODEL_SELECTION` to your model_name.
3. Navigate to [pipeline.py](src/pipeline.py) and add the relevant imports for your model under the `Models` segment.
4. Scroll all the way down to the `MODEL PIPELINES` segment and start a new `elif` for your model. A template for the pipeline has been put in place and commented out. Remember to implement 2 pipelines (`tuning_pipeline`, `model_pipeline`).

#### HYPERPARAMETERS

To tune the hyperparameters required for your model, the steps are as follows:

1. Navigate to [config.yml](src/config.yml) and scroll down to the `HYPERPARAMETERS` segment.
2. Locate the model you wish to change the hyperparameters for, they are label in the following format: `MODELNAME_DTYPES_HPARAMS`. You can add or change the values as follows, just make sure to do it under the correct dtypes.
3. If you wish to add hyperparameters configurations for your NEW models, proceed to step 4 after completing the appropriate additon of configurations here.
4. (ONLY FOR NEW CONFIGURATIONS) Navigate to [core.py](src/config/core.py) and locate the `ModelConfig` class. Scroll down to the `HYPERPARAMS` segment and add accordingly CONFIGURATION_NAME and the datatype. To be strictly adhered to or there will be an error.

## d. Description of logical steps/flow of the pipeline

### 1. processing_pipeline

1. Basic Cleaning: Fixed negative values, format string issues and ensure consistent null values where applicable.
2. Various Imputations: Different imputations for different columns of data
3. Feature Creation: Create Engineered features from the initial data columns
4. Drop Features: Drop features that are not meaningful or identified as possible candidates for multicollinearity.

### 2. encoding_pipeline

1. OneHotEncode Nominal Categorical variable such that they can be used by the models
2. OrdinalEncode Ordinal Categorical variable to retain order in the variables

### 3. feature_selection_pipeline

1. Drop duplicate features, which are identified as columns with the exact same values for all rows
2. Drop features that might exhibit multicollinearity using variance as the metric

The above 3 pipelines are all consolidated in `total_pipeline`, which can be further customized with the optional `check_pipeline`, which purpose it for checking the dataframe content and info.

### 4. model_pipeline/tuning_pipeline

1. Preprocessed the data via the steps above
2. Fit the selected model on the training data

## **e. Overview of key findings from the EDA**

### Preprocessing and Findings

1. Out of the 10348 rows initially in the dataset, there were 350 repeated ids from `"ID"`. Further investigation into those rows revealed that the entire row data is the duplicated with another row. As such these rows are to be dropped and we will have 9998 rows.
2. Further investigation into the features, it is discovered that some values in the `"Age"` column are negative. These values are likely to be keyed wrongly and hence was corrected via taking absolute of those values.
3. For the `"Gender"` column, there were a mixture of similar values with different string formatting (I.E: "Male" and "MALE"), and hence will be flagged out as 2 different values if fixed. Appropriate formatting is done to ensure consistent values across.
4. A few columns namely `"COPD History"`, `"Taken Bronchodilators"`, `"Air Pollution Exposure"` and `"Gender"` were detected to have missing values. Further investigation were completed on those columns in the EDA notebook and followed up with appropriate imputations.
5. Target Variable (`Lung Cancer Occurrence`) distribution was approximately similar, with just a very minor skew towards "negative".
6. Weights columns (`Last Weight` and `Current Weight`) were found to have some sort of relationship and further identified as good candidates for feature engineering.
7. For the categorical variables, `Genetic Markers` and `Air Pollution Exposure` were found to be important variables affecting the target variable.
8. Validation of data into the columns `Start Smoking` and `Age` found inconsistencies where a portion of the patients apparently started smoking from when they are 0 years old!
9. `Start Smoking` and `Stop Smoking` are ambiguous columns of data by themselves, thus I widely considered them as ideal candidate for feature engineering as well.

## **f.  Describe how the features in the dataset are processed (summarised in a table).**

| Feature | Type | Processed |
| - | - | - |
| ID | Numerical | Omitted |
| Age | Numerical | Take absolute value. Impute Median if required |
| Gender | Categorical | Formatted. Impute Mode if required |
| COPD History | Categorical | ReferenceImputed (refer to feateng.py) |
| Genetic Markers | Categorical | Impute Mode if required |
| Air Pollution Exposure | Categorical | Impute Mode if required |
| Last Weight | Numerical | ReferenceImputed (refer to feateng.py). Omitted after feature_engineered |
| Current Weight | Numerical | ReferenceImputed (refer to feateng.py). Omitted after feature_engineered |
| Start Smoking | String | Omitted after feature_engineered |
| Stop Smoking | String | Omitted after feature_engineered |
| Taken Bronchodilators | Categorical | ReferenceImputed (refer to feateng.py). Omitted when feature_selection on duplicate is run |
| Frequency of Tiredness | Categorical | Impute Mode if required |
| Dominant Hand | Categorical | Impute Mode if required |
| Lung Cancer Occurrence | Categorical | Target Variable |
| Change in Weight | Numerical | Engineered_feature from "Last Weight" and "Current Weight". |
| Years Smoked | Numerical | Engineered_feature from "Start Smoking" and "Stop Smoking". Omitted after testing with and without. |
| Cat Smoker | Categorical | Engineered_feature from "Start Smoking" and "Stop Smoking". |

## **g. Explanation of your choice of models for each machine learning task.**

3 models were used for evaluation, SVC (sklearn), MLPClassifier (sklearn), LightGradientBoostingMachine (lightgbm)

1. SVC:
   - Support Vector machine is particularly efficient in this case as we are dealing with a small dataset and thus the amount of training time required is manageable. As it has relatively lesser hyperparameters and a clear decision boundary determined by the training data, it is less likely to overfit to the training data and generalizes well.

2. MLPClassifier:
   - Neural networks are known to perform quite well in classification tasks and are highly customisable to size and n_features of the dataset we are looking at. MLPClassifier in scikit-learn contain options for regularization which is appropriate here as it was identified during EDA that quite a number of columns have little significance with reference to the target variable.

3. LightGradientBoostingMachine:
   - In the dataset that we have, there are quite a number of categorical variables, which LGBM can actually handled without any encoding required. In the pipeline I did proceed with encoding in the preprocessing as it is a generalized pipeline for all models, but in a separate case we can customized the pipeline differently. Like MLPClassifier, it offers regularization which is relatively helpful in this case if we can identify the few important features.

## **h. Evaluation of the models developed.**

- SVC
  - Recall - 71.20%
  - F1 Score - 69.13%
- MLPClassifier
  - Recall - 86.23%
  - F1 Score - 76.16%
- LightGradientBoostingMachine
  - Recall - 82.61%
  - F1 Score - 77.22%
 
Recall is used as I want my model to reduce False Positives as much as possible since Lung Cancer treatment or further procedures are rather expensive, it will be a huge expense if there are many false positives. F1_Score on the other hand, other than accounting for False Positives as well, it looks at False Negatives. Lung Cancer if detected early can be treated and managed better, thus I would want to reduce the amount of misses in this case if possible such that they can be detected at early stages before they spread or worsen.

## **i. Other considerations for deploying the models developed.**

TODO: API Implementation of training/predicting where the model trains on the whole dataset and have predict function to take a testfile as input and make predictions on that file.

TODO: Further selection/filter of features for training the model

TODO: Error handling mechanisms for possible data issues that are not present in this data source.
