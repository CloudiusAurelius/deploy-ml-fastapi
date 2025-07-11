# Script to train machine learning model.

from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference
import pandas as pd
import logging
import pickle
import argparse


# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

logger.info("Starting model training script")


# Parse command line arguments
parser = argparse.ArgumentParser(description="Model training.")
parser.add_argument(
    "--grid_search", "-g",
    help="Grid search for hyperparameters",
    action="store_true",
    default=False,
)
args = parser.parse_args()

logging.info(f"Grid search enabled: {args.grid_search}")


# Define the categorical features and label column.
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

label_column = "salary"



# Add code to load in the data.
logger.info("Loading data")
data = pd.read_csv("data/census_cleaned.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
logger.info("Splitting data into training and test sets")
train, test = train_test_split(data, test_size=0.20)
logger.info("Data split completed")

# Log the number of rows and columns in the training and test data.
train_n_rows = train.shape[0]
train_n_columns = train.shape[1]
test_n_rows = test.shape[0]
test_n_columns = test.shape[1]
logger.info(f"Training data:\
            number of rows: {train_n_rows},\
            number of columns: {train_n_columns}")
logger.info(f"Test data:\
            number of rows: {test_n_rows},\
            number of columns: {test_n_columns}")



# Process the training data with the process_data function.
logger.info("Processing training data")
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label=label_column, training=True
)

# Proces the test data with the process_data function.
logger.info("Processing test data")    
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label=label_column,
    training=False,
    encoder=encoder,
    lb=lb,
)

# Train and save a model.
logger.info("Training model")
model = train_model(X_train, y_train, grid_search=args.grid_search)
# Save training date and time
current_datetime = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

# Compute the model's predictions and metrics.
logger.info("Computing model metrics")
y_pred = inference(model, X_test)
precision,\
recall,\
fbeta,\
roc_auc = compute_model_metrics(y_test, y_pred)
logger.info(f"Model metrics:\
            Precision: {precision:.4f},\
            Recall: {recall:.4f},\
            F-beta: {fbeta:.4f},\
            ROC AUC: {roc_auc:.4f}")
# Save traing metrics to a text file.
logger.info("Saving model metrics to text file")
with open("./model/model_metrics_training.txt", "w") as filehandler:
    filehandler.write(f"Precision: {precision:.4f}\n")
    filehandler.write(f"Recall: {recall:.4f}\n")
    filehandler.write(f"F-beta: {fbeta:.4f}\n")
    filehandler.write(f"ROC AUC: {roc_auc:.4f}\n")

# Save the model to a file.
# --------------------------------------
logger.info("Saving model to file")
# Create a dictionary with model information.
model_info = {
    "name": "logistic_regression_model",
    "created_at": current_datetime,
    "model": model,
    "params": model.get_params(),
    "precision": precision,
    "recall": recall,
    "fbeta": fbeta,
    "roc_auc": roc_auc,
    "rows_train": train_n_rows,
    "columns_train": train_n_columns,
    "rows_test": test_n_rows,
    "columns_test": test_n_columns,
    "categorical_features": cat_features,
    "label_column": label_column,    
}
logging.info(f"Saving model with model info: {model_info}")

# Save model with model info to a pickle file.
with open("./model/log_reg_model.pkl", "wb") as filehandler:
    pickle.dump(model_info, filehandler)
logger.info("Model exported")


# Save the encoder to a file.
logger.info("Saving encoder to file")
with open("./model/encoder.pkl", "wb") as filehandler:
    pickle.dump(encoder, filehandler)
logger.info("Encoder exported")

# Save the label binarizer to a file.
logger.info("Saving label binarizer to file")
with open("./model/label_binarizer.pkl", "wb") as filehandler:
    pickle.dump(lb, filehandler)
logger.info("Label binarizer exported")
