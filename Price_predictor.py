import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error
import matplotlib.pyplot as plt

# load data
data = fetch_california_housing()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# no data cleaning since no Nan values in dataset
# creating a basline for evaluating models later
scaler = StandardScaler()
cv = KFold(n_splits=5, shuffle=True, random_state=2)

base_pipeline = Pipeline(
    steps=[
        ('preprocessor', scaler),
        ('dummyregressor', DummyRegressor())
    ]
)
base_scores = -cross_val_score(base_pipeline, X, y, cv=cv, scoring='neg_root_mean_squared_error')
print(f"baseline rmse scores with dummy regressor:\nmean: {base_scores.mean()}\nstd: {base_scores.std()}")

# creating 2 different pipelines to evaluate best model
# forest_pipeline = Pipeline(
#     steps=[
#         ('preprocessor', scaler),
#         ('forestmodel', RandomForestRegressor(random_state=2))
#     ]
# )
gradient_boost_pipeline = Pipeline(
    steps=[
        ('preprocessor', scaler),
        ('histmodel', HistGradientBoostingRegressor(random_state=2))
    ]
)

# tuning hyperparameters on RandomizedSearchCV to reduce time
# forest_params = {
#     "forestmodel__n_estimators": range(100, 500, 100),
#     "forestmodel__max_depth": [10, 20, 30, 40, None],
#     "forestmodel__min_samples_leaf": range(1, 5),
#     "forestmodel__max_features": [(num/4 + 0.25) for num in range(0, 3)]
# }
# forest_search = RandomizedSearchCV(forest_pipeline, param_distributions=forest_params, scoring="neg_root_mean_squared_error", 
#                                    random_state=2, cv=cv, n_iter=15, n_jobs=-1)
# forest_search.fit(X_train, y_train)
# forest_model = forest_search.best_estimator_

gradient_boost_params = {
    "histmodel__max_iter": range(200, 600, 100),
    "histmodel__learning_rate": [num/100 for num in range(3, 20, 3)],
    "histmodel__max_depth": range(3, 6),
    "histmodel__min_samples_leaf": [10, 20, 40],
    "histmodel__l2_regularization": [0.0, 0.1, 1.0],
}
gradient_search = RandomizedSearchCV(gradient_boost_pipeline, param_distributions=gradient_boost_params, scoring="neg_root_mean_squared_error",
                                     random_state=2, cv=cv, n_iter=15, n_jobs=-1)
gradient_search.fit(X_train, y_train)
gradient_boost_model = gradient_search.best_estimator_

# print(-forest_search.best_score_, forest_search.best_params_)
print(-gradient_search.best_score_, gradient_search.best_params_)

# histgradientboostregressor was the better model after tuning hyperparameters
# results
pred = gradient_boost_model.predict(X_test)
print(root_mean_squared_error(y_test, pred))