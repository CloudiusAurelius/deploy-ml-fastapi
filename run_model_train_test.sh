#!/bin/bash
# This script runs the model training and evaluation process.

# Set the environment variable for the model directory
export MODEL_DIR="./model"  


# Clean the input dataecho 
echo "-----------------------------------------------------"
echo "Starting data cleaning..."
echo "-----------------------------------------------------"
echo "Executing code in ./data/clean_data.py"
python ./data/clean_data.py\
    --input_file ./data/census.csv\
    --output_file ./data/census_cleaned.csv


# Train the model
echo -e "\n-----------------------------------------------------"
echo "Starting model training..."
echo "-----------------------------------------------------"
echo "Executing code in ./ml/model.py"
python ./train_model.py --grid_search


# Run unit tests
echo -e "\n-----------------------------------------------------"
echo "Running unit tests..."
echo "-----------------------------------------------------"
echo "Executing code in ./tests/functions_tests.py"
PYTHONPATH=$(pwd) pytest -v ./tests/unit_tests.py


# Evaluate the model on slices
echo -e "\n-----------------------------------------------------"
echo "Evaluating model on slices..."
echo "-----------------------------------------------------"
echo "Executing code in ./evaluate_model.py"
python ./evaluate_model.py