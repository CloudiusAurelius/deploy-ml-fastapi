import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from model import train_model
from tests.model_tests import evaluate_slices
#import joblib
import pickle
from ml.data import process_data
from ml.model import compute_model_metrics, inference
import logging

def evaluate_slices(data, feature, model, encoder, label_column):
    """
    Evaluate model performance on slices of data for a categorical feature.

    Parameters
    ----------
    data : pd.DataFrame
        The full dataset including categorical and label columns.
    feature : str
        The name of the categorical feature to slice on.
    model : sklearn model
        Trained model for inference.
    encoder : fitted OneHotEncoder or similar
        Used to transform categorical data.
    label_column : str
        Name of the column with true labels.

    Returns
    -------
    results : dict
        Dictionary mapping feature values to metric tuples (precision, recall, fbeta).
    """
    results = {}

    for value in data[feature].unique():
        slice_df = data[data[feature] == value]
        if slice_df.empty:
            continue

        X_slice = slice_df.drop(columns=[label_column])
        y_slice = slice_df[label_column]

        # Encode if necessary
        X_encoded = encoder.transform(X_slice)
        preds = inference(model, X_encoded)

        metrics = compute_model_metrics(y_slice, preds)
        results[value] = metrics

    return results

if __name__ == "__main__":


    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
    logger = logging.getLogger()

    # Load the data
    logger.info("Loading data")
    df = pd.read_csv("data/census_cleaned.csv")
    logger.info("Data loaded successfully")
   
    # Define the label column
    # Note: Ensure this column exists in your DataFrame
    # If the column is not present, you may need to adjust the DataFrame accordingly.
    label_column = "salary"
    if label_column not in df.columns:
        raise ValueError("The label column 'salary' is not present in the DataFrame.")
    
       
    
    # Define categorical features and label column
    cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
    ]
    
    
    # Load the trained model, encoder and label binarizer
    logger.info("Loading model")        
    with open("./model/log_reg_model.pkl", "rb") as filehandler:
        model = pickle.load(filehandler)    
    logger.info("Model loaded successfully")
    
    # Load the encoder
    logger.info("Loading encoder")
    with open("./model/encoder.pkl", "rb") as filehandler:  
        encoder = pickle.load(filehandler)
    logger.info("Loaded model encoder successfully")

    # Load the label binarizer
    logger.info("Loading label binarizer")
    with open("./model/label_binarizer.pkl", "rb") as filehandler:
        lb = pickle.load(filehandler)
    logger.info("Loaded label binarizer successfully")


    # Process data in evalution mode
    X_train, y_train, encoder, lb = process_data(
        X=df,
        categorical_features=cat_features,
        label=label_column,
        training=False,
        encoder=encoder,
        lb=lb
    )


    # Evaluate slices for each categorical feature    
    logger.info("Starting evaluation of model slices")
    for feature in cat_features:
        logger.info(f"Evaluating slices for feature: {feature}")
        evaluate_slices(
            data=df,
            feature=feature,
            model=model,
            encoder=encoder,
            label_column=label_column
        )
        logger.info(f"Completed evaluation for feature: {feature}")     
       
    logger.info("Evaluation completed")
  