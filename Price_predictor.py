import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor
import matplotlib.pyplot as plt

data = fetch_california_housing()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

base_pipeline = Pipeline(steps=[('preprocessor', StandardScaler()), ('dummyregressor', DummyRegressor())])
cv = KFold(n_splits=5, shuffle=True, random_state=2)
base_scores = -cross_val_score(base_pipeline, X, y, cv=cv, scoring='neg_root_mean_squared_error')
print(f"baseline rmse scores with dummy regressor:\nmean: {base_scores.mean()}\nstd: {base_scores.std()}")