from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_get_path():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"message": "Welcome to the API!"}


def test_post_path0():
    r = client.post("/predict",
        json={
            "age": 39,
            "workclass": "State-gov",
            "fnlgt": 77516,
            "education": "Bachelors",
            "education_num": 13,
            "marital_status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "capital_gain": 2174,
            "capital_loss": 0,
            "hours_per_week": 40,
            "native_country": "United-States"
        }
    )
    assert r.status_code == 200
    assert r.json() == {
        "input_data": {
            "age": 39,
            "workclass": "State-gov",
            "fnlgt": 77516,
            "education": "Bachelors",
            "education_num": 13,
            "marital_status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "capital_gain": 2174,
            "capital_loss": 0,
            "hours_per_week": 40,
            "native_country": "United-States"
        },
        "prediction": [
            " <=50K"
        ]
    }


def test_post_path1():
    r = client.post("/predict",
        json={
            "age": 31,
            "workclass": "Private",
            "fnlgt": 45781,
            "education": "Masters",
            "education_num": 14,
            "marital_status": "Never-married",
            "occupation": "Prof-specialty",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Female",
            "capital_gain": 14084,
            "capital_loss": 0,
            "hours_per_week": 50,
            "native_country": "United-States"
        }
    )
    assert r.status_code == 200
    assert r.json() == {
        "input_data": {
            "age": 31,
            "workclass": "Private",
            "fnlgt": 45781,
            "education": "Masters",
            "education_num": 14,
            "marital_status": "Never-married",
            "occupation": "Prof-specialty",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Female",
            "capital_gain": 14084,
            "capital_loss": 0,
            "hours_per_week": 50,
            "native_country": "United-States"
        },
        "prediction": [
            " >50K"
        ]
    }












def test_get_path():
    r = client.get("/items/42")
    assert r.status_code == 200
    assert r.json() == {"fetch": "Fetched 1 of 42"}


def test_get_path_query():
    r = client.get("/items/42?count=5")
    assert r.status_code == 200
    assert r.json() == {"fetch": "Fetched 5 of 42"}


def test_get_malformed():
    r = client.get("/items")
    assert r.status_code != 200