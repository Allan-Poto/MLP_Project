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

## d. [Description of logical steps/flow of the pipeline](src/pipeline.py)

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

## **f. Explanation of your choice of models for each machine learning task.**

3 models were used for evaluation, SVC (sklearn), MLPClassifier (sklearn), LightGradientBoostingMachine (lightgbm)

1. SVC:
   - 
2. MLPClassifier:
   - 
3. LightGradientBoostingMachine:
   - 

## **g. Evaluation of the models developed.**

- SVC
  - Recall - 
  - F1 Score - 
- MLPClassifier
  - Recall - 
  - F1 Score - 
- LightGradientBoostingMachine
  - Recall - 
  - F1 Score - 
