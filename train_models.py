import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import joblib

df = pd.read_csv("houses.csv")


def build_linear_model():
    X = df[["size", "nb_rooms", "garden"]]
    y = df["price"]
    model = LinearRegression()
    model.fit(X, y)
    joblib.dump(model, "linear.joblib")


def build_logistic_model():
    X = df[["size", "nb_rooms", "garden"]]
    y = np.where(df["price"] < 0, -1, np.where(df["price"] < 270000, 0, 1))
    model = LogisticRegression()
    model.fit(X, y)
    joblib.dump(model, "logistic.joblib")


def build_decision_tree_model():
    X = df[["size", "nb_rooms", "garden"]]
    y = df["price"] < 270000
    # y = np.where(df["price"] < 0, -1, np.where(df["price"] < 270000, 0, 1))
    model = DecisionTreeClassifier()
    model.fit(X, y)
    joblib.dump(model, "tree.joblib")


build_linear_model()
build_logistic_model()
build_decision_tree_model()
