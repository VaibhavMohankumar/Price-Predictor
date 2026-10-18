import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# load data
data = fetch_california_housing(as_frame=True)
X, y = data.data, data.target
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

scaler = StandardScaler()
cv = KFold(n_splits=10, shuffle=True, random_state=2)

# creating a basline for evaluating models later

base_pipeline = Pipeline(
    steps=[
        ('preprocessor', scaler),
        ('dummyregressor', DummyRegressor())
    ]
)
base_scores = -cross_val_score(base_pipeline, X, y, cv=cv, scoring='neg_root_mean_squared_error')
print(f"Baseline RMSE scores with Dummy Regressor:\nmean: {base_scores.mean()}\nstd: {base_scores.std()}")

# creating 2 different pipelines to evaluate best model
# forest_pipeline = Pipeline(
#     steps=[
#         ('preprocessor', scaler),
#         ('forestmodel', RandomForestRegressor(random_state=2))
#     ]
# )

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
    "histmodel__max_iter": [300, 500, 800, 1200],
    "histmodel__learning_rate": [num/100 for num in range(3, 20, 3)],
    "histmodel__max_depth": range(3, 9),
    "histmodel__min_samples_leaf": [10, 20, 25, 30, 40],
    "histmodel__l2_regularization": [0.0, 0.1, 0.2, 1.0]
}

# function to run model with and without feature engineering
def run(feature_fn=None):
    steps = []
    if feature_fn:
        steps.append(('features', FunctionTransformer(feature_fn)))
    steps.append(('histmodel', HistGradientBoostingRegressor(random_state=2)))
    
    # tuning hyperparameters on RandomizedSearchCV to reduce time
    search = RandomizedSearchCV(Pipeline(steps), param_distributions=gradient_boost_params, scoring='neg_root_mean_squared_error', random_state=2,
                                cv=cv, n_iter=100, n_jobs=-1)
    
    return search.fit(X_train, y_train)

def fold_rmse(search):
    i = search.best_index_
    return -np.array([search.cv_results_[f'split{k}_test_score'][i] for k in range(cv.get_n_splits())])

before = run()
after = run(add_features)

# positive = features helped
diff = fold_rmse(before) - fold_rmse(after)

print("Hist Gradient Boosting Regressor before/after diff")
print(f"mean gain {diff.mean():.4f} ± {diff.std()/np.sqrt(len(diff)):.4f} (SE)")
print(f"{(diff > 0).sum()} of {len(diff)} folds improved")

for name, s in [('before', before), ('after', after)]:
    print(name, 'test RMSE:', round(root_mean_squared_error(y_test, s.predict(X_test)), 3))

# histgradientboostregressor was the better model after tuning hyperparameters
# results
pred_before = before.predict(X_test)
pred_after = after.predict(X_test)
y_true = np.asarray(y_test)
capped = y_true >= 4.99

print("In depth test results after feature engineering")
print("MAE:", mean_absolute_error(y_test, pred_after), "| R²:", r2_score(y_test, pred_after))
print("Baseline test RMSE:", root_mean_squared_error(y_test, np.full(len(y_test), y_train.mean())))
print(f"Capped rows: {capped.sum()} ({capped.mean():.1%})")
print("RMSE excluding capped rows:", root_mean_squared_error(y_true[~capped], pred_after[~capped]))

df = X_test.assign(actual=y_true, predicted=pred_after, error=y_true - pred_after)
worst = df.loc[df['error'].abs().nlargest(10).index]
print(worst[['MedInc', 'HouseAge', 'Latitude', 'Longitude', 'actual', 'predicted', 'error']])
print(after.best_params_)

# residual calculations
residuals_before = y_true - pred_before
residuals_after = y_true - pred_after


def plot_diff(pred, residuals, title, figure_title):
    _, axes = plt.subplots(1, 3, figsize=(18, 5))
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

    plt.suptitle(f'Hist Gradient Boosting: Test Set Performance {title} Feature Engineering', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{figure_title}.png', bbox_inches='tight')
    
plot_diff(pred_before, residuals_before, "before", "predictions_before")
plot_diff(pred_after, residuals_after, "after", "predictions_after")

plt.show()