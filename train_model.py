# Script to train machine learning model.

from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference
import pandas as pd
import logging
#import joblib
import pickle


# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


# Add code to load in the data.
data = pd.read_csv("data/census_cleaned.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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

# Process the training data with the process_data function.
logger.info("Processing training data")
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
logger.info("Processing test data")    
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# Train and save a model.
logger.info("Training model")
model = train_model(X_train, y_train, grid_search=True)

# Compute the model's predictions and metrics.
logger.info("Computing model metrics")
y_pred = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, y_pred)

# Save the model to a file.
#joblib.dump(model, "log_reg_model.pkl")
with open("./model/log_reg_model.pkl", "wb") as filehandler:
    pickle.dump(model, filehandler)
logger.info("Model exported")

# Save the encoder to a file.
#joblib.dump(encoder, "encoder.pkl")
with open("./model/encoder.pkl", "wb") as filehandler:
    pickle.dump(encoder, filehandler)
logger.info("Encoder exported")

# Save the label binarizer to a file.
#joblib.dump(lb, "label_binarizer.pkl")
with open("./model/label_binarizer.pkl", "wb") as filehandler:
    pickle.dump(lb, filehandler)
logger.info("Label binarizer exported")
