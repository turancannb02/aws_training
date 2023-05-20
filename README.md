# aws_training

# Predicting Power Consumption with Machine Learning

## Introduction

This project focuses on applying machine learning techniques to predict the power consumption in a specific zone. The predictive model is developed using a linear regression model and is trained on historical power consumption data. The aim is to create a robust model that can effectively predict future power consumption, which can aid in the planning and optimization of energy resources.

## Data Description

The data used in this project comes from a CSV file named 'powerconsumption.csv'. Each row represents an hour, and the columns represent the date-time of the observation and the power consumption of Zone 1. The dataset is a time series, and the task is to predict the power consumption of Zone 1. 

## Tools and Libraries Used

- Python: The project is implemented in Python due to its flexibility and the availability of several suitable libraries for data processing and machine learning.
- pandas: Used for data manipulation and analysis.
- numpy: Used for numerical computations.
- matplotlib: Used for data visualization.
- sklearn: Used for machine learning tasks including model building, model evaluation, and preprocessing.
- joblib: Used for saving the model for future use.

## Project Structure / Files Description

The project consists of a single Python script named 'aws_train.py' that includes all the steps from loading the data to training the model and evaluating it. 

## How to Run the Project

To run the project, follow these steps:

1. Ensure that you have all the necessary Python libraries installed (pandas, numpy, matplotlib, sklearn, joblib).
2. Download the 'powerconsumption.csv' dataset and make sure it's in the same directory as the 'aws_train.py' script.
3. Run the 'aws_train.py' script using a Python interpreter.

## Results / Conclusion

The project resulted in a linear regression model that predicts the power consumption with a Root Mean Squared Error of 522.12 on the test set. The model was also validated using cross-validation, resulting in high R-squared scores across all folds, which indicates a good fit of the model to the data. The model, along with the analysis and feature engineering, can help to make better predictions about power consumption, which can be used to optimize energy management and planning.
