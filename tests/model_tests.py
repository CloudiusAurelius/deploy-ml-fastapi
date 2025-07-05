import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from ml.model import train_model, compute_model_metrics, inference

# Create test data
X_train = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
y_train = np.array([0, 1, 1, 0])
X_test = np.array([[0, 1], [1, 1]])
y_test = np.array([0, 1])


def test_train_model_default():
    model = train_model(X_train, y_train, grid_search=False)
    assert isinstance(model, LogisticRegression)
    assert hasattr(model, "predict")
    assert model.classes_.tolist() == [0, 1]


def test_inference_output_shape():
    model = train_model(X_train, y_train)
    preds = inference(model, X_test)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (2,)
    assert set(preds).issubset({0, 1})


def test_compute_model_metrics():
    precision, recall, f1 = compute_model_metrics(y_test, y_test)
    assert precision == 1.0
    assert recall == 1.0
    assert f1 == 1.0

    y_pred = np.array([1, 0])
    precision, recall, f1 = compute_model_metrics(y_test, y_pred)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(f1, float)


def test_train_model_with_grid_search():
    model = train_model(X_train, y_train, grid_search=True)
    assert isinstance(model, LogisticRegression)
    assert hasattr(model, "predict")




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

