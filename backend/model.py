import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# Load data
df = pd.read_csv('../data/sales.csv')

# Convert Date
df['Date'] = pd.to_datetime(df['Date'])

# Select useful columns
df = df[['Date', 'Product line', 'Unit price', 'Quantity', 'Total']]

# Feature Engineering
df['day'] = df['Date'].dt.day
df['month'] = df['Date'].dt.month
df['day_of_week'] = df['Date'].dt.dayofweek

# Convert categorical to numerical
df = pd.get_dummies(df, columns=['Product line'])

# Features & Target
X = df.drop(['Total', 'Date'], axis=1)
y = df['Total']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
error = mean_absolute_error(y_test, predictions)

print("Model trained successfully!")
print("Mean Absolute Error:", error)



joblib.dump(model, 'sales_model.pkl')
print("Model saved!")