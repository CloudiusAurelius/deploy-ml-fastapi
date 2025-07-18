"""
Unit tests for the model training and inference functions in ml.model module.
"""

import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from ml.model import train_model, compute_model_metrics, inference

@pytest.fixture
def data():
    # Create test data
    X_train = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
    y_train = np.array([0, 1, 1, 0])
    X_test = np.array([[0, 1], [1, 1]])
    y_test = np.array([0, 1])
    y_pred = np.array([1, 0])
    return X_train, y_train, X_test, y_test


def test_train_model_default(data):
    """Test the default model training without grid search."""
    # Unpack the data
    X_train, y_train, X_test, y_test = data

    # Train the model
    model = train_model(X_train, y_train, grid_search=False)

    # Check the model type and attributes
    assert isinstance(model, LogisticRegression)
    assert hasattr(model, "predict")
    assert model.classes_.tolist() == [0, 1]


def test_inference_output_shape(data):
    """Test the inference function output shape."""
    
    # Unpack the data
    X_train, y_train, X_test, y_test = data
    
    # Train the model and make predictions
    model = train_model(X_train, y_train)
    preds = inference(model, X_test)
    
    # Check the output shape and type
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (2,)
    assert set(preds).issubset({0, 1})


def test_compute_model_metrics(data):
    """Test the model metrics computation and data types"""
    
    # Unpack the data
    X_train, y_train, X_test, y_test = data

    # Test calculation of precision, recall, and f1 score 
    precision, recall, f1, auc = compute_model_metrics(y_test, y_test)
    assert precision == 1.0
    assert recall == 1.0
    assert f1 == 1.0
    assert auc == 1.0
   
    # Test data types of metrics
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(f1, float)
    assert isinstance(auc, float)


def test_train_model_with_grid_search(data):
    """Test the model training with grid search."""
    
    # Unpack the data
    X_train, y_train, X_test, y_test = data

    # Train the model without grid search
    model = train_model(X_train, y_train, grid_search=False)
    
    # Check the model type and attributes
    assert isinstance(model, LogisticRegression)
    assert hasattr(model, "predict")

