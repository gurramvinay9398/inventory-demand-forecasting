import pandas as pd
import joblib
from datetime import timedelta

# Load model
model = joblib.load('demand_model.pkl')

# Load dataset
df = pd.read_csv('../data/train.csv')
df['date'] = pd.to_datetime(df['date'])

last_date = df['date'].max()

# 🔥 USER INPUT
store = int(input("Enter store number: "))
item = int(input("Enter item number: "))
days = int(input("Enter number of days to predict (1-10): "))

future_data = []

for i in range(1, days + 1):
    future_date = last_date + timedelta(days=i)

    future_data.append({
        'store': store,
        'item': item,
        'day': future_date.day,
        'month': future_date.month,
        'day_of_week': future_date.dayofweek,
        'week': future_date.isocalendar().week
    })

future_df = pd.DataFrame(future_data)

# Prediction
predictions = model.predict(future_df)

future_df['date'] = [last_date + timedelta(days=i) for i in range(1, days+1)]
future_df['predicted_demand'] = predictions

print(future_df[['date', 'predicted_demand']])