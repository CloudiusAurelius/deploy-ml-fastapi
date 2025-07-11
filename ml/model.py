from sklearn.metrics import fbeta_score,\
    precision_score,\
    recall_score,\
    roc_auc_score
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


# Optional: implement hyperparameter tuning.
def train_model(
    X_train,
    y_train,
    grid_search=False,
    param_grid = [
        {'solver': ['liblinear'], 'penalty': ['l2'], 'C': [1, 10]},
        {'solver': ['saga'], 'penalty': ['l2'], 'C': [1]}
    ]
    ):
    """   
    This function creates a logistic regression model, fits it to the training data,
    and returns the trained model. If `grid_search` is set to True, it can be extended
    to include hyperparameter tuning.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    grid_search : bool
        Whether to perform hyperparameter tuning using grid search.   

    Returns
    -------
    model
        Trained machine learning model.
    """
    # Create a logistic regression model
    model = LogisticRegression(max_iter=4000)
    
    # Tuned hyperparameters can be added here if needed
    # For example, you can use GridSearchCV or RandomizedSearchCV for hyperparameter tuning
    if grid_search:        
        print(f"Performing grid search for hyperparameter tuning with param_grid: {param_grid}")
        
        # Create a grid search object
        grid = GridSearchCV(model, param_grid, cv=5, scoring='f1')

        # Fit the grid search to the training data
        grid.fit(X_train, y_train)

        model = grid.best_estimator_
        print("Best parameters found:", grid.best_params_)
        print("Best score from grid search:", grid.best_score_)

    else:
        # Fit the model to the training data
        model.fit(X_train, y_train)
        print("Model training completed.")
    
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    roc_auc = roc_auc_score(y, preds)
    return precision, recall, fbeta, roc_auc


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : 
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    # Use the model to make predictions
    preds = model.predict(X)

    # Return the predictions
    return preds
