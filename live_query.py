# This script performs a live query on the deployed API.

# Libraries
import requests
import json

# Define the url
url = 'https://deploy-ml-fastapi.onrender.com'

# Data to be sent in the POST request
data = {
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


# Make the POST request to the API
response = requests.post(\
    url+'/predict', data=json.dumps(data)
)
    
# Check if the request was successful
if response.status_code == 200:
    # Parse the JSON response
    prediction = response.json()
    print("Prediction:", prediction)
else:
    print("Error:", response.status_code, response.text)