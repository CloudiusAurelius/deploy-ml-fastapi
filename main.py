#------------------------------
# Contains the code of the API.
# -----------------------------
from fastapi import FastAPI
from pydantic import BaseModel, Field

import pickle
import pandas as pd
import logging

from ml.data import process_data
from ml.model import inference


# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger(__name__)

# Instantiate the FastAPI app
logger.info("Instantiating FastAPI app...")
app = FastAPI()
logger.info("FastAPI app instantiated")


# Declare a Pydantic model for the input data
class InputData(BaseModel):
    """
    Pydantic model for input data.
    This model defines the structure of the data expected in the prediction request.
    """
    age: int = Field(
        default=39,
        examples=[39],
        description="Age of the individual"
    )
    workclass: str = Field(
        default="State-gov",
        examples=["State-gov"],
        description="Type of work class"
    )
    fnlgt: int = Field(
        default=77516,
        examples=[77516]
    )
    education: str = Field(
        default="Bachelors",
        examples=["Bachelors"],
        description="Education level"
    )
    education_num: int = Field(
        default=13,
        examples=[13],
        description="Number of years of education"
    )
    marital_status: str = Field(
        default="Never-married",
        examples=["Never-married"], 
        description="Marital status of the individual"
    )
    occupation: str = Field(
        default="Adm-clerical",
        examples=["Adm-clerical"],
        description="Occupation of the individual"
    )
    relationship: str = Field(
        default="Not-in-family",
        examples=["Not-in-family"],
        description="Relationship status of the individual"
    )
    relationship: str = Field(
        default="Not-in-family",
        examples=["Not-in-family"],
        description="Relationship status of the individual"
    )
    race: str = Field(
        default="White",
        examples=["White"]
    )        
    sex: str = Field(
        default="Male",
        examples=["Male"]
    )
    capital_gain: int = Field(
        default=2174,
        examples=[2174],
        description="Capital gain of the individual"
    )
    capital_loss: int = Field(
        default=0,
        examples=[0],
        description="Capital loss of the individual"
    )
    hours_per_week: int = Field(
        default=40,
        examples=[40],
        description="Hours worked per week"
    )
    native_country: str = Field(
        default="United-States",
        examples=["United-States"],
        description="Country of origin"
    )               
 
   
  
# Load the model, econder, and categorical features on startup
@app.on_event("startup")
def load_model():
    logger.info("Loading model and components on startup...")
    try:
        with open("./model/log_reg_model.pkl", "rb") as filehandler:
            app.state.model = pickle.load(filehandler)["model"]        
        with open("./model/encoder.pkl", "rb") as filehandler:  
            app.state.encoder = pickle.load(filehandler)
        with open("./model/label_binarizer.pkl", "rb") as filehandler:
            app.state.lb = pickle.load(filehandler)
        app.state.categorical_features = [
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",    
        ]
    except Exception as e:
        logger.error(f"Error loading model or components: {e}")
        raise RuntimeError("Startup failed: could not\
            load model or components") from e

# Define a GET on the specified endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the ML model API!"}


@app.post("/predict")
async def predict(data: InputData):
    """
    Endpoint to make predictions with the ML model.
    :param data: Input data for prediction.
    :return: Dictionary with the input data and the prediction result.
             The "prediction" field is a list containing the predicted class label(s) for the input.
    """
    # Load the model and make a prediction
    # This is a placeholder for the actual model loading and prediction logic.
    # In a real application, you would load your trained model here and use it to make predictions.

    # Load the model from a pickle file
    # ----------------------------------------------------
    # Load the trained model, encoder, and label binarizer
    # ----------------------------------------------------
    
    
    # Load the model from the app state
    model = app.state.model
    encoder = app.state.encoder
    binarizer = app.state.lb
    categorical_features = app.state.categorical_features

    # Aggregate the input data into a DataFrame
    df_api = pd.DataFrame([data.dict()])
    df_api = df_api.rename(
        columns={
            'education_num': 'education-num',
            'marital_status': 'marital-status',
            'native_country': 'native-country',
            'capital_gain': 'capital-gain',
            'capital_loss': 'capital-loss',
            'hours_per_week': 'hours-per-week'
        }
    )


    # Process the input data to obtain numpy arrays for inference
    # Note: 
    #   - y is empty (inference mode)  
    #   - label binarizer is not used in inference mode
    X_api, _, _, _ = process_data(
        X=df_api, # Input DataFrame
        categorical_features=categorical_features, # Categorical features defined in the app state
        label=None, # No label column in inference mode
        training=False, # Inference mode
        encoder=encoder, # Use the encoder from the app state
        lb=None # No label binarizer in inference mode
    )    

    # Prediction using the model
    preds = inference(model, X_api)

    # Convert predictions to human-readable format
    human_readable_preds = binarizer.inverse_transform(preds)

    
    return {
        "input_data": data.dict(),  # Return the input data as a dictionary
        "prediction": human_readable_preds.tolist()  # Convert numpy array to list for JSON serialization
    }
    
