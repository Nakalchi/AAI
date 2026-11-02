python -m pip install scikit-learn pandas fastapi uvicorn joblib numpy

import pandas as pd
import numpy as np
import joblib

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

from fastapi import FastAPI
from pydantic import BaseModel



data = load_iris()

X = data.data
y = data.target



model = RandomForestClassifier(random_state=42)

model.fit(X, y)



joblib.dump(model, "iris_model.pkl")

print("Model saved as iris_model.pkl")



app = FastAPI()


model = joblib.load("iris_model.pkl")



class IrisRequest(BaseModel):

    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.post("/predict/")
def predict(iris: IrisRequest):

    
    features = np.array([[
        iris.sepal_length,
        iris.sepal_width,
        iris.petal_length,
        iris.petal_width
    ]])

 
    prediction = model.predict(features)

    species = [
        "setosa",
        "versicolor",
        "virginica"
    ]

    return {
        "prediction": species[prediction[0]]
    }



