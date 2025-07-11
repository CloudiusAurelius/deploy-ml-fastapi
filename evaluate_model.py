import pandas as pd
import numpy as np
import pickle
from ml.data import process_data
from ml.model import compute_model_metrics, inference
import logging

def evaluate_slices(data, categorical_features, feature, model, encoder, label_binarizer, label_column):
    """
    Evaluate model performance on slices of data for a categorical feature.

    Parameters
    ----------
    data : pd.DataFrame
        The full dataset including categorical and label columns.
    categorical_features : list[str]
        List of categorical feature names. Encoding is applied to these features.
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
        # Slice the DataFrame for the current feature value
        # Note: This assumes that the feature is categorical and has a finite set of values.
        logging.info(f"Selecting slice of the data based on feature '{feature}' with value '{value}'")
        slice_df = data[data[feature] == value]
        # Store the number of rows and columns in the slice
        logging.info(f"Slice shape for feature '{feature}' with value '{value}': {slice_df.shape}")        
        if slice_df.empty:
            logging.warning(f"Skipping slice for feature '{feature}' with value '{value}' due to empty data.")  
            continue # Skip empty slices

        # Process the data frame in evalution mode
        logging.info(f"Data processing slice for feature '{feature}' with value '{value}'")
        X_slice, y_slice, encoder, lb = process_data(
            X=slice_df,
            categorical_features=categorical_features,
            label=label_column,
            training=False,
            encoder=encoder,
            lb=label_binarizer
        )    

        # Predict using the model
        # Note: Ensure that the model is compatible with the processed data.
        #if X_slice.empty or y_slice.empty:
        if X_slice.size == 0 or y_slice.size == 0:
            logging.warning(f"Skipping slice for feature '{feature}' with value '{value}' due to empty data.")  
            continue # Skip empty slices        
        preds = inference(model, X_slice)

        # Compute metrics
        # Note: Ensure that the compute_model_metrics function is compatible with the model's output.
        if len(np.unique(y_slice)) == 1:
            # If there's only one class in y_slice, skip this slice
            logging.warning(f"Skipping slice for feature '{feature}' with value '{value}' due to single class in y_slice.") 
            continue
        metrics = compute_model_metrics(y_slice, preds)
        
        # Store the results
        results[value] = metrics
        logging.info(f"Evaluated slice for feature '{feature}' with value '{value}': {metrics}")

    return results


def store_textfile(feature, results_feature, outfilepath):
    """
    Store the evaluation results for a specific feature in a text file.

    Parameters
    ----------
    feature : str
        The name of the categorical feature.
    results_feature : dict
        Dictionary mapping feature values to metric tuples (precision, recall, fbeta).    
    """
    with open(outfilepath, "w") as filehandler:
        for value, metrics in results_feature.items():                
            precision, recall, fbeta, auc = metrics
            filehandler.write(f"Feature: {feature},\
                    Value: {value},\
                    Precision: {precision:.4f},\
                    Recall: {recall:.4f},\
                    F-beta: {fbeta:.4f},\
                    AUC: {auc:.4f}\n")

if __name__ == "__main__":

    # ----------------------------------------------------
    # Set up logging and variables
    # ----------------------------------------------------

    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
    logger = logging.getLogger()

    logger.info("Starting model evaluation script")


    # Load the data
    logger.info("Loading data")
    df = pd.read_csv("data/census_cleaned.csv")
    logger.info("Data loaded successfully")
   
    # Define the label column
    # Note: Ensure this column exists in your DataFrame
    # If the column is not present, you may need to adjust the DataFrame accordingly.
    label_column = "salary"
    logger.info(f"Label column: {label_column}")
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
    logger.info(f"Categorical features: {cat_features}")
    
    # ----------------------------------------------------
    # Load the trained model, encoder, and label binarizer
    # ----------------------------------------------------

    # Load the trained model, encoder and label binarizer
    logger.info("Loading model")        
    with open("./model/log_reg_model.pkl", "rb") as filehandler:
        model_info = pickle.load(filehandler)
    model = model_info["model"]
    model_name = model_info["name"]
    model_created_at = model_info["created_at"]
    model_params = model_info["params"]
    logger.info(f"Model '{model_name}' loaded successfully")
    logger.info(f"Model created at: {model_created_at}")
    logger.info(f"Model parameters: {model_params}")
    
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



    # ----------------------------------------------------
    # Evaluate slices for each categorical feature
    # ----------------------------------------------------
    # This will iterate over each categorical feature and evaluate the model's performance on slices of the data.
    # The results will be saved in text files for each feature.
    # ----------------------------------------------------    
       
    
    logger.info("Starting evaluation of model slices")
    for feature in cat_features:
        logger.info(f"Evaluating slices for feature: {feature}")
        # Evaluate slices for the current feature
        results_feature = evaluate_slices(
            data=df,
            categorical_features=cat_features,
            feature=feature,
            model=model,
            encoder=encoder,
            label_binarizer=lb,
            label_column=label_column
        )
        logger.info(f"Completed evaluation for feature: {feature}")     

        # Store results in text file        
        outfilepath = f"./model/slice_results_{feature}.txt"
        store_textfile(feature, results_feature, outfilepath)
        logger.info(f"Results for feature '{feature}' saved to {outfilepath}.\n\n")
               
    logger.info("Evaluation completed")
  