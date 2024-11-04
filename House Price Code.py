


# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from feature_engine.outliers import Winsorizer

# Loading data
data = pd.read_csv(r"C:\Users\Piyus Pahi\Documents\ML Project Prodigy Infotech\Data Set\realest.csv")
print(data.head())

# Database connection (optional, skip if not needed)
from sqlalchemy import create_engine
engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
                       .format(user="root", pw="965877", db="house_db"))
data.to_sql('house', con=engine, if_exists='replace', index=False)

# Exploratory Data Analysis
data.info()
print(data.describe())

# Dropping unwanted columns
data = data.drop(["Condition"], axis=1)

# Handling missing values and scaling numeric features
num_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='mean')),
    ('scale', MinMaxScaler())
])
clean_data = num_pipeline.fit_transform(data)
cleandata = pd.DataFrame(clean_data, columns=data.columns)
print(cleandata.isnull().sum())

# Outlier treatment with Winsorization
winsor = Winsorizer(capping_method='iqr', tail='both', fold=1.5,
                    variables=["Price", "Bedroom", "Space", "Room", "Lot", "Tax", "Bathroom", "Garage"])
cleandata[["Price", "Bedroom", "Space", "Room", "Lot", "Tax", "Bathroom", "Garage"]] = winsor.fit_transform(
    cleandata[["Price", "Bedroom", "Space", "Room", "Lot", "Tax", "Bathroom", "Garage"]])
print(cleandata.describe())

# Splitting data into features (X) and target (y)
X = cleandata[['Bedroom', 'Space', 'Room', 'Lot', 'Tax', 'Bathroom', 'Garage']]
y = cleandata['Price']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling data for improved prediction
scaler = StandardScaler()
x_train_std = scaler.fit_transform(x_train)
x_test_std = scaler.transform(x_test)

# Trying Polynomial Features (degree 2)
poly_features = PolynomialFeatures(degree=2, include_bias=False)
x_train_poly = poly_features.fit_transform(x_train_std)
x_test_poly = poly_features.transform(x_test_std)

# Model 1: Linear Regression with Polynomial Features
poly_lr = LinearRegression()
poly_lr.fit(x_train_poly, y_train)
y_pred_poly_lr = poly_lr.predict(x_test_poly)
print("R2 Score with Polynomial Linear Regression:", r2_score(y_test, y_pred_poly_lr))

# Model 2: Ridge Regression with Polynomial Features
ridge = Ridge(alpha=1.0)
ridge.fit(x_train_poly, y_train)
y_pred_ridge = ridge.predict(x_test_poly)
print("R2 Score with Ridge Regression:", r2_score(y_test, y_pred_ridge))

# Model 3: Lasso Regression with Polynomial Features
lasso = Lasso(alpha=0.01)
lasso.fit(x_train_poly, y_train)
y_pred_lasso = lasso.predict(x_test_poly)
print("R2 Score with Lasso Regression:", r2_score(y_test, y_pred_lasso))

# Model 4: Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)
y_pred_rf = rf_model.predict(x_test)
print("R2 Score with Random Forest Regressor:", r2_score(y_test, y_pred_rf))

# Model 5: Gradient Boosting Regressor
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
gb_model.fit(x_train, y_train)
y_pred_gb = gb_model.predict(x_test)
print("R2 Score with Gradient Boosting Regressor:", r2_score(y_test, y_pred_gb))

# Choosing the best model based on R2 Score
models = {
    "Polynomial Linear Regression": r2_score(y_test, y_pred_poly_lr),
    "Ridge Regression": r2_score(y_test, y_pred_ridge),
    "Lasso Regression": r2_score(y_test, y_pred_lasso),
    "Random Forest Regressor": r2_score(y_test, y_pred_rf),
    "Gradient Boosting Regressor": r2_score(y_test, y_pred_gb)
}
best_model = max(models, key=models.get)
print(f"Best Model: {best_model} with R2 Score: {models[best_model]}")

import joblib

# Save the best model
joblib.dump(gb_model, 'best_model.pkl')  # Here, replace `gb_model` with your best model

import os
os.getcwd()
