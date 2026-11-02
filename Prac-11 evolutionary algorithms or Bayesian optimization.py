python -m pip install numpy pandas scikit-learn scikit-optimize

import numpy as np
import pandas as pd
import joblib

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from skopt import BayesSearchCV
from sklearn.metrics import accuracy_score, classification_report



data = load_iris()
X, y = data.data, data.target



X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


model = RandomForestClassifier(random_state=42)



param_space = {
    "n_estimators": (10, 200),
    "max_depth": (1, 20),
    "min_samples_split": (2, 10),
    "min_samples_leaf": (1, 10),
    "max_features": ["sqrt", "log2", None]
}


optimizer = BayesSearchCV(
    estimator=model,
    search_spaces=param_space,
    n_iter=30,
    cv=3,
    random_state=42,
    n_jobs=1  
)


print("Starting Bayesian Optimization...")

optimizer.fit(X_train, y_train)



best_model = optimizer.best_estimator_

y_pred = best_model.predict(X_test)


print("\nBest Parameters:")
print(optimizer.best_params_)

print("\nAccuracy on Test Set:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


joblib.dump(best_model, "optimized_rf_model.pkl")

print("\nModel saved as 'optimized_rf_model.pkl'.")
