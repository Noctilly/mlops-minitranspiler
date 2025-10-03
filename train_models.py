import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
import joblib

df = pd.read_csv("houses.csv")


def build_linear_model():
    X = df[["size", "nb_rooms", "garden"]]
    y = df["price"]
    model = LinearRegression()
    model.fit(X, y)
    joblib.dump(model, "regression.joblib")


def build_logistic_model():
    X = df[["size", "nb_rooms", "garden"]]
    y = np.where(df["price"] < 0, -1, np.where(df["price"] < 270000, 0, 1))
    model = LogisticRegression()
    model.fit(X, y)
    joblib.dump(model, "regression.joblib")


build_logistic_model()
