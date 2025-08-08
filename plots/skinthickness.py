from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV

import pandas as pd
import joblib
import numpy as np

# Load your dataset
df = pd.read_csv("data\diabetes.csv")

# Use only rows with valid SkinThickness
df = df[df['SkinThickness'] > 0]

# Features and target
features = ['BMI', 'Insulin', 'BloodPressure']
X = df[features]
y = df['SkinThickness']

# Train-test split (e.g., 80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=3,                # Limit tree depth
    min_samples_split=10,       # Require more samples to split
    min_samples_leaf=5,         # Don't allow tiny leaves
    random_state=42
)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"R² score: {r2:.3f}")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

y_pred = model.predict(X_train)

# Evaluation metrics
mse = mean_squared_error(y_train, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_train, y_pred)
r2 = r2_score(y_train, y_pred)

print(f"R² score: {r2:.3f}")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

joblib.dump(model, 'skinthickness_model.pkl')
