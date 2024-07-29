import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

# Load the dataset
df = pd.read_csv('Fish.csv')

# Data preprocessing
# df['Species'] = pd.factorize(df['Species'])[0]
X = df.iloc[:, 2:].values
y = df.iloc[:, 1].values
# print(X)
# print(y)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'mean Squared Error: {mse}')

# Save the model
joblib.dump(model, 'fish_weight_model.pkl')
