import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from model import train_model
from evaluate import evaluate_slices

def test_evaluate_slices(df):
    # Encode categorical and numeric features
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    X = encoder.fit_transform(df[['feature', 'num']])
    y = df['label'].values

    # Train model
    model = train_model(X, y)

    # Evaluate on slices
    results = evaluate_slices(df, feature='feature', model=model, encoder=encoder, label_column='label')

    assert isinstance(results, dict)
    for k, v in results.items():
        assert isinstance(k, str)
        assert len(v) == 3  # precision, recall, fbeta
        assert all(isinstance(metric, float) for metric in v)

if __name__ == "__main__":
    df = pd.read_csv('data/slice_test_data.csv')

    test_evaluate_slices()
    print("All tests passed!")