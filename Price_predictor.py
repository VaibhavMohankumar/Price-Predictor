import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline, FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error
import matplotlib.pyplot as plt

# load data
data = fetch_california_housing(as_frame=True)
X, y = data.data, data.target
print(X.columns)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# no data cleaning since no Nan values in dataset

# feature engineering to derive better insights from data

# approximate (lat, lon) of major cities
cities = {
    'SF': (37.77, -122.42), 'LA': (34.05, -118.24), 'SD': (32.72, -117.16),
    'SJ': (37.34, -121.89), 'Sac': (38.58, -121.49)
}
def add_features(X):
    X = X.copy()
    
    for city, (lat, long) in cities.items():
        # calculate distance of house to every major city
        X[f'dist_{city}'] = np.hypot((X['Latitude'] - lat), (X['Longitude'] - long))
    # Pick the closest distance to a major city
    X['major_city_min_dist'] = X[[f'dist_{city}' for city in cities]].min(axis=1)
    
    # calculate average bedroom ratio
    X["bedroom_ratio"] = X['AveBedrms'] / X['AveRooms']

    # transform population data to reduce skew
    X['log_pop'] = np.log1p(X['Population'])
    return X

# Pipeline for feature engineering

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
        ('features', FunctionTransformer(add_features)),
        ('preprocessor', scaler),
        ('histmodel', HistGradientBoostingRegressor(random_state=2))
    ]
)

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

# tuning hyperparameters on RandomizedSearchCV to reduce time
gradient_boost_params = {
    "histmodel__max_iter": range(200, 10000, 100),
    "histmodel__learning_rate": [num/100 for num in range(3, 20, 3)],
    "histmodel__max_depth": range(3, 6),
    "histmodel__min_samples_leaf": [10, 20, 25, 30, 40],
    "histmodel__l2_regularization": [0.0, 0.1, 0.2, 1.0]
}
gradient_search = RandomizedSearchCV(gradient_boost_pipeline, param_distributions=gradient_boost_params, scoring="neg_root_mean_squared_error", random_state=2, cv=cv, n_iter=100, n_jobs=-1)
gradient_search.fit(X_train, y_train)
gradient_boost_model = gradient_search.best_estimator_

# print(-forest_search.best_score_, forest_search.best_params_)
print(-gradient_search.best_score_, gradient_search.best_params_)

# histgradientboostregressor was the better model after tuning hyperparameters
# results
pred = gradient_boost_model.predict(X_test)

y_true = np.asarray(y_test)
residuals = y_true - pred

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Predicted vs actual
axes[0].scatter(y_true, pred, alpha=0.3, s=10)
lims = [0, max(y_true.max(), pred.max())]
axes[0].plot(lims, lims, color='red', linestyle='--', label='Perfect prediction')
axes[0].set_xlabel('Actual value ($100k)')
axes[0].set_ylabel('Predicted value ($100k)')
axes[0].set_title('Predicted vs Actual')
axes[0].legend()

# 2. Residuals vs predicted
axes[1].scatter(pred, residuals, alpha=0.3, s=10)
axes[1].axhline(0, color='red', linestyle='--')
axes[1].set_xlabel('Predicted value ($100k)')
axes[1].set_ylabel('Residual (actual - predicted)')
axes[1].set_title('Residuals vs Predicted')

# 3. Distribution of residuals
axes[2].hist(residuals, bins=50, edgecolor='black')
axes[2].axvline(0, color='red', linestyle='--')
axes[2].set_xlabel('Residual ($100k)')
axes[2].set_ylabel('Count')
axes[2].set_title('Residual Distribution')

plt.suptitle('Hist Gradient Boosting: Test Set Performance', fontsize=14)
plt.tight_layout()
plt.savefig('predictions_initial.png', bbox_inches='tight')
plt.show()