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

# Validate input
if store < 1 or store > 10:
    print("Invalid store! Choose between 1 and 10")
    exit()

if item < 1 or item > 50:
    print("Invalid item! Choose between 1 and 50")
    exit()

if days < 1 or days > 10:
    print("Days must be between 1 and 10")
    exit()

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
predictions = predictions.round().astype(int)

future_df['date'] = [last_date + timedelta(days=i) for i in range(1, days+1)]
future_df['predicted_demand'] = predictions

print(future_df[['date', 'predicted_demand']])

# 🔥 USER INPUT (stock)
current_stock = int(input("Enter current stock: "))

# Total predicted demand
total_demand = future_df['predicted_demand'].sum()

# Safety stock factor
recommended_stock = int(total_demand * 1.2)

print("\nTotal Predicted Demand:", total_demand)
print("Recommended Stock:", recommended_stock)

# 🚨 Alerts
if current_stock < total_demand:
    print("🚨 ALERT: Low Stock! Reorder needed")
elif current_stock > recommended_stock:
    print("⚠️ ALERT: Overstock! Reduce inventory")
else:
    print("✅ Stock level is optimal")