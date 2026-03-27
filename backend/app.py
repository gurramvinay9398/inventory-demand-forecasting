from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
from datetime import timedelta
import os 
from model import train_model      

app = Flask(__name__)
CORS(app)

MODEL_PATH = "demand_model.pkl"
if os.path.exists(MODEL_PATH):
    model = joblib.load('demand_model.pkl')
else:
    model=train_model()


@app.route('/')
def home():
    return "Sales Prediction API Running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        store = int(data.get("store"))
        item = int(data.get("item"))
        days = int(data.get("days"))
        current_stock = int(data.get("stock"))

        # Validation
        if store < 1 or store > 10:
            return jsonify({"error": "Store must be 1-10"})
        if item < 1 or item > 50:
            return jsonify({"error": "Item must be 1-50"})
        if days < 1 or days > 10:
            return jsonify({"error": "Days must be 1-10"})

        
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        DATA_PATH = os.path.join(BASE_DIR, 'data', 'train.csv')
        print("Loading data from:", DATA_PATH)

        df = pd.read_csv(DATA_PATH, sep=',', engine='python')

        df.columns = df.columns.str.strip().str.lower()

        print("Columns:", df.columns)

        if 'date' not in df.columns:
            raise ValueError("Date column missing")
        
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

        # 🔥 Convert using explicit parsing fallback
        if df['date'].isna().sum() > 0:
            df['date'] = pd.to_datetime(df['date'].astype(str), format='%Y-%m-%d', errors='coerce')

        if df['date'].isna().sum() > 0:
            df['date'] = pd.to_datetime(df['date'].astype(str), format='%m/%d/%Y', errors='coerce')

        # Drop invalid rows
        df = df.dropna(subset=['date'])
        print("NaT count:", df['date'].isna().sum())
        print("After cleaning shape:", df.shape)

        df = df.sort_values(by='date', kind='stable')

        last_date = df['date'].max()

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

        X_future = future_df[['store', 'item', 'day', 'month', 'day_of_week', 'week']]

        predictions = model.predict(X_future)
        predictions = predictions.round().astype(int)

        future_df['predicted_demand'] = predictions

        total_demand = int(predictions.sum())
        recommended_stock = int(total_demand * 1.2)

        # Alerts
        if current_stock < total_demand:
            alert = "Low Stock"
        elif current_stock > recommended_stock:
            alert = "Overstock"
        else:
            alert = "Stock Optimal"

        return jsonify({
            "daily_predictions": future_df[['predicted_demand']].to_dict(orient='records'),
            "total_demand": total_demand,
            "recommended_stock": recommended_stock,
            "alert": alert
        })

    except Exception as e:
        return jsonify({"error": str(e)})
    
@app.route('/analytics', methods=['POST'])
def analytics():
    
    data = request.json

    days = int(data.get("days", 7))
    top_items_n = int(data.get("top_items", 5))
    top_stores_n = int(data.get("top_stores", 5))

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, 'data', 'train.csv')
    print("Loading data from:", DATA_PATH)
    
    df = pd.read_csv(DATA_PATH, sep=',', engine='python')

    df.columns = df.columns.str.strip().str.lower()

    print("Columns:", df.columns)

    if 'date' not in df.columns:
        raise ValueError("Date column missing")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # 🔥 Convert using explicit parsing fallback
    if df['date'].isna().sum() > 0:
        df['date'] = pd.to_datetime(df['date'].astype(str), format='%Y-%m-%d', errors='coerce')

    if df['date'].isna().sum() > 0:
        df['date'] = pd.to_datetime(df['date'].astype(str), format='%m/%d/%Y', errors='coerce')

    # Drop invalid rows
    df = df.dropna(subset=['date'])
    print("NaT count:", df['date'].isna().sum())
    print("After cleaning shape:", df.shape)

    df = df.sort_values(by='date', kind='stable')

    

    # 📈 Demand trend
    trend = df.groupby('date')['sales'].sum().tail(days)

    # 🏬 Store performance
    store = df.groupby('store')['sales'].sum().sort_values(ascending=False).head(top_stores_n)

    # 🔥 Top items
    items = df.groupby('item')['sales'].sum().sort_values(ascending=False).head(top_items_n)

    return {
        "trend_labels": trend.index.strftime('%Y-%m-%d').tolist(),
        "trend_data": trend.values.tolist(),

        "store_labels": store.index.astype(str).tolist(),
        "store_data": store.values.tolist(),

        "item_labels": items.index.astype(str).tolist(),
        "item_data": items.values.tolist()
    }


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)