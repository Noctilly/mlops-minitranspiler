import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
import joblib
import c_functions

df = pd.read_csv("houses.csv")


model: LogisticRegression = joblib.load("regression.joblib")

print(model.coef_)
print(model.classes_)
print(df[["size", "nb_rooms", "garden"]].iloc[[0]])
print(model.predict_proba(df[["size", "nb_rooms", "garden"]].iloc[[0]]))


def linear_model_to_c(model: LinearRegression):
    return f"""
    #include <stdio.h>
          
    {c_functions.linear_regression()}
    
    void main(){{
        double thetas[{len(model.coef_) + 1}] = {{ {", ".join([str(model.intercept_)] + [str(coef) for coef in model.coef_])} }};
        int n_parameters = {len(model.coef_) + 1};

        double features[3] = {{ 205.9991686803, 2, 0 }};

        printf("Prediction: %f", linear_regression_prediction(features, thetas, n_parameters));
    }}"""


def logistic_model_to_c(model: LogisticRegression):
    features = [205.9991686803, 2, 0]

    main_func = (
        c_functions.multiclasses_logistic_regression
        if len(model.classes_) > 2
        else c_functions.binary_logistic_regression
    )

    return f"""
    #include <stdio.h>
    
    {c_functions.pow()}
    
    {c_functions.factorial()}
    
    {c_functions.exp_aprox()}
          
    {c_functions.sigmoid()}
          
    {c_functions.logistic_regression()}
    
    {main_func(model=model, features=features)}
"""


print(logistic_model_to_c(model=model))
