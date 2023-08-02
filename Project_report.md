# Report: Predict Bike Sharing Demand with AutoGluon Solution
#### NAME HERE

## Initial Training
### What did you realize when you tried to submit your predictions? What changes were needed to the output of the predictor to submit your results?
During my initial presentation of predicted results on Kaggle, a lot of changes needed to be made before the presentation. But as usual this was my first test in which the model was trained with default parameters as the baseline model to get only one score so I can use it to compare with the predicted score after doing feature engineering and hyperparameter tuning I can use Initially I only parse the datetime column of the dataset, which is not enough for better performance of the model. Here are some of the parameters and data columns I used in my initial prediction:

* Since 'count' is the target column I'm trying to estimate, this is the label I set.
* I ignore the 'random' and 'registered' columns as they are not present in the test dataset.
* I choose `root_mean_squared_error` as the metric to be used for evaluation.
* I have set a time limit of 10 minutes (600 seconds).
* And the preset I use to focus on building the best model is `best_quality`.

Since I used the data directly without doing feature engineering, the predictive score of the model was not that good and the score on Kaggle was also not that good.

### What was the top ranked model that performed?
The top ranked model in the initial training is **WeightedEnsemble_L3.** The following are the various results of this model:

* Model: The model is a weighted combination model, which combines the predictions of several individual models to produce the final prediction.

* score_value: The validation score for this model is **-50.739530**.

* pred_time_val: The prediction time for the validation set is **5.291339 seconds.** It indicates the time taken by the model to make predictions on the validation data.

* fit_time: The fit time represents the total time taken by the model to train, including all individual models and ensemble builds. In this case, the fit time is **421.446217 seconds.**

* can_infer: A value of **True** indicates that the model is capable of making predictions.

## Exploratory data analysis and feature creation
### What did the exploratory analysis find and how did you add additional features?
Exploratory analysis is the preliminary examination of data to discover patterns, relationships, and insights. It is often used to understand the structure of a dataset and identify possible relationships between variables.

* The data for `year`, `month`, `day`, `day of the week` and `hour` were extracted as independent features separate from the `datetime` feature using data feature extraction. After feature extraction, the `datetime` feature was removed.

* Feature `datetime` was parsed as a datetime feature to retrieve the hour information from the timestamp

* The independent attributes `season` and `weather` were initially read as `integer`. Since these are categorical variables, they were converted to the `category` data type.

* The `casual` and `registered` features are only present in the train dataset and absent in the test data, therefore, these features were ignored during model training  and after removing there, it was observed that the RMSE scores improved significantly and these independent features were highly correlated with the target variable.

### How much better did your model preform after adding additional features and why do you think that is?
Analyzing the best model scores of the `initial prediction` and the `second prediction (after EDA)`, it is observed that there is an improvement of `74%`.

According to me this is possible because of the following changes made to the data.

* I have extracted the `year`, `month`, `day`, `day of week` and `hour` elements of the datetime feature as individual attributes. which leads to a significant improvement in the predicted values.

* Also the `Data Type` of the attributes `Season` and `Weather` are changed to categorical variables. Which also contributes to this improvement.

* And last thing I am not considering features like `casual` and `registered` in `training set` because they are not present in `test data set`.

## Hyper parameter tuning
### How much better did your model preform after trying different hyper parameters?
Analyzing the best model scores of the `initial prediction` and the `third prediction`, it is observed that there is an improvement of `75%`.

### If you were given more time with this dataset, where do you think you would spend more time?
In fact I already spend more time in forecasting with hyperparameter and I have seen a little more improvement compared to the second forecast. Therefore, I would prefer to spend more time in the third prediction (with hyperparameters).

### Create a table with the models you ran, the hyperparameters modified, and the kaggle score.
import pandas as pd
pd.DataFrame({
    "model": ["initial", "add_features", "hpo"],
    "hpo1": ["eval_matrix: 'root mean squared error'", "perset: best quality", "Time Limit: 600"],
    "hpo2": ["eval_matrix: 'root mean squared error'", "perset: best quality", "Time Limit: 600"],
    "hpo3": ["auto_stack=True", "refit_full=True", "Time Limit: 1200"],
    "score": [1.7999, 0.45138, 0.44362]
})
### Create a line plot showing the top model score for the three (or more) training runs during the project.

![model_train_score.png](img/model_test_score.png)

### Create a line plot showing the top kaggle score for the three (or more) prediction submissions during the project.

![model_test_score.png](img/model_test_score.png)

## Summary
Automated Stack Ensemble and Regression Models: AutoGluon was used to create both automated stack ensembles and individual regression models for tabular data. This approach allowed for quick prototyping of baseline models, saving significant time and effort in the initial stages of the project.

**Improved Results with EDA and Feature Engineering:** The top-ranked AutoGluon-based model demonstrated significant improvement by leveraging data obtained after extensive exploratory data analysis (EDA) and feature engineering. This highlights the importance of data preprocessing and understanding the data's underlying patterns.

**Automatic Hyperparameter Tuning and Model Selection:** AutoGluon's capabilities in automatic hyperparameter tuning, model selection, and ensembling proved to be beneficial. It allowed the framework to explore and exploit the best possible combinations of hyperparameters and models, leading to enhanced performance without manual intervention.

**Improved Performance through Hyperparameter Tuning:** Utilizing AutoGluon's hyperparameter tuning resulted in improved performance compared to the initial raw submission. This indicates that fine-tuning model hyperparameters can have a significant impact on predictive accuracy.

**Cumbersome Process and Time Limit Dependency:** One observation was that hyperparameter tuning using AutoGluon could be a cumbersome process. The success of hyperparameter tuning was found to be highly dependent on factors like time limits, prescribed presets, the family of models used, and the range of hyperparameters to be tuned. This suggests that finding the optimal hyperparameters can be a challenging task and may require experimentation.

Overall, it's evident that AutoGluon's capabilities in automating various aspects of the machine learning pipeline, such as hyperparameter tuning and model selection, greatly contributed to the success of the bike sharing demand prediction project. However, it's important to acknowledge that hyperparameter tuning can be a complex and time-consuming process, and finding the right balance between exploration and exploitation can be crucial for achieving the best possible results.
