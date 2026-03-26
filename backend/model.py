import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# Load data
df = pd.read_csv('../data/train.csv')

# Convert date
df['date'] = pd.to_datetime(df['date'])

# Feature Engineering (VERY IMPORTANT 🔥)
df['day'] = df['date'].dt.day
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek
df['week'] = df['date'].dt.isocalendar().week

# Features & Target
X = df[['store', 'item', 'day', 'month', 'day_of_week', 'week']]
y = df['sales']

# Time-based split (IMPORTANT)
df = df.sort_values('date')

split = int(len(df) * 0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

# Model
model = RandomForestRegressor(n_estimators=20, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

# Evaluation
error = mean_absolute_error(y_test, pred)

print("MAE:", error)


joblib.dump(model, 'demand_model.pkl')
print("Model saved!")