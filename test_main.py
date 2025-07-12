from fastapi.testclient import TestClient

from main import app




def test_get_path():
    with TestClient(app) as client: # Ensures the app is running before creating a test client        
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "Welcome to the ML model API!"}


def test_post_path0():    
    with TestClient(app) as client: # Ensures the app is running before creating a test client
        response = client.post("/predict",
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
        assert response.status_code == 200
        assert "prediction" in response.json()
        assert response.json() == {
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
    with TestClient(app) as client: # Ensures the app is running before creating a test client
        
        response = client.post("/predict",
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
        assert response.status_code == 200
        assert "prediction" in response.json()
        assert response.json() == {
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




