import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import joblib
import c_functions

df = pd.read_csv("houses.csv")


model: DecisionTreeClassifier = joblib.load("tree.joblib")

print(model.tree_.feature)
print(model.tree_.children_left)
print(model.tree_.children_right)
print(model.tree_.threshold)
print(model.tree_.value)
print(model.classes_)
print(df[["size", "nb_rooms", "garden"]].iloc[[0]])
print(model.predict(df[["size", "nb_rooms", "garden"]].iloc[[0]]))
print(model.predict_proba(df[["size", "nb_rooms", "garden"]].iloc[[0]]))


def linear_model_to_c(model: LinearRegression):
    features = [205.9991686803, 2, 0]
    return f"""
    #include <stdio.h>
          
    {c_functions.linear_regression()}
    
    {c_functions.linear_regression_main(model=model, features=features)}    
    """


def logistic_model_to_c(model: LogisticRegression):
    features = [205.9991686803, 2, 0]

    main_func = (
        c_functions.multiclasses_logistic_regression_main
        if len(model.classes_) > 2
        else c_functions.binary_logistic_regression_main
    )

    return f"""
    #include <stdio.h>
    
    {c_functions.pow()}
    
    {c_functions.factorial()}
    
    {c_functions.exp_aprox()}
          
    {c_functions.sigmoid()}

    {c_functions.linear_regression()}
          
    {c_functions.logistic_regression()}
    
    {main_func(model=model, features=features)}
"""


def decision_tree_model_to_c(model: DecisionTreeClassifier):
    input = [205.9991686803, 2, 0]

    return f"""
    #include <stdio.h>

    {c_functions.decision_tree_classifier()}

    {c_functions.decision_treee_classifier_main(model, input)}
"""


print(decision_tree_model_to_c(model=model))
