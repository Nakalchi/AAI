
# install the package in cmd python -m pip install pandas numpy scikit-learn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Load Dataset
# Using the Titanic dataset for demonstration
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
data = pd.read_csv(url)

# Step 2: Basic Preprocessing
# Drop unnecessary columns
data = data.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

# Handle missing values
data["Age"] = data["Age"].fillna(data["Age"].median())
data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])

# Separate features and target
X = data.drop("Survived", axis=1)
y = data["Survived"]

# Step 3: Split Data into Train and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Define Preprocessing Steps
# Numerical features: Scale values
numerical_features = ["Age", "Fare"]
numerical_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

# Categorical features: One-hot encode
categorical_features = ["Sex", "Embarked", "Pclass"]
categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Combine preprocessors into a column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Step 5: Define Feature Selection and Model Options
feature_selection = SelectKBest(score_func=f_classif)

# Define candidate models
models = {
    "RandomForest": RandomForestClassifier(random_state=42),
    "SVC": SVC(random_state=42)
}

# Step 6: Create the Pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("feature_selection", feature_selection),
    ("classifier", RandomForestClassifier())
])

# Step 7: Define Grid Search for Hyperparameter Tuning
# (one grid per model, since n_estimators only applies to RandomForest and C only to SVC)
param_grid = [
    {
        "feature_selection__k": [5, 6, 7],
        "classifier": [models["RandomForest"]],
        "classifier__n_estimators": [100, 200]
    },
    {
        "feature_selection__k": [5, 6, 7],
        "classifier": [models["SVC"]],
        "classifier__C": [0.1, 1, 10]
    }
]
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring="accuracy", verbose=2)

# Step 8: Train the Model
grid_search.fit(X_train, y_train)

# Step 9: Evaluate the Model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("Best Parameters:", grid_search.best_params_)
print("Accuracy on Test Set:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
