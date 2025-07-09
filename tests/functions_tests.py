import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from ml.model import train_model, compute_model_metrics, inference


# Unit tests for the model training and evaluation functions
# Assuming X_train, y_train, X_test, y_test are defined globally


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





if __name__ == "__main__":
    
    # Create test data
    X_train = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
    y_train = np.array([0, 1, 1, 0])
    X_test = np.array([[0, 1], [1, 1]])
    y_test = np.array([0, 1])
    
    # Run the tests
    test_train_model_default()
    test_inference_output_shape()
    test_compute_model_metrics()
    test_train_model_with_grid_search()

    # Result
    print("All tests passed!") 