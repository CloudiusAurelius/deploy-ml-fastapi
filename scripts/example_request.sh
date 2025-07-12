curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
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
}'



"""
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
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
}'
"""
