# Model Card

## Model Details
The model was created by Bernd Galler. It was derived by using Logistic Regression with hyperparemeter tuning. The tuning was performed over parameters such as regularization strength and penalty type.

## Intended Use
This model should be used to predict salary based on census data. 

## Training Data
The was trained on a labeled data set (label "salary" - binary: >50K, <=50K) of Census Bureau data containing 26048 observations including categorical and numberical variables. This corresponds to 80% of the available data. Data preprocessing included encoding categorical features. 

## Evaluation Data
The model was evaluated on a hold-out test set of 6513 observations (20% of the total data). Performance was measured on metrics such as accuracy, precision, recall, F1-score and AUC to ensure balanced evaluation accross classes.


## Metrics
The following metrics were recorded during model training:

Precision: 0.7094

Recall: 0.2477

F1-score: 0.3672

AUC (Area Under ROC Curve): 0.6084

These metrics indicate that the model's performance is moderate on the overall sample.

In addition, the performance was evaluated on several data slices. For this purpose, a subset of the data was selected for each value of the categorical features. This analysis rendered that the model performance is balanced for most categories. Single deviations from the overall performance can be detected, but it has to be noted that the observation number is note always high in every subcategory (e.g., native-country="Peru" with AUC: 0.75 and 31 observations or native-country="Hungary" with AUC: 0.45 and 13 observations).

## Ethical Considerations
Potential biases in the training data may affect model fairness, especially regarding. It is recommended to continuously monitor the model’s predictions in production for fairness and unintended biases. In regard to the major categories such as sex an overall balanced model performance is obtained (e.g., AUC female: 0.6126 vs AUC male: 0.6284).

## Caveats and Recommendations
The model’s performance depends heavily on the quality and representativeness of the training data.

It may not generalize well to data distributions significantly different from the training set.

Hyperparameter tuning was limited to logistic regression parameters; further tuning or alternative models might improve performance.

Regular retraining and validation are recommended to maintain accuracy over time.
