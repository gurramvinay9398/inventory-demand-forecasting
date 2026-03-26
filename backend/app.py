from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
from datetime import timedelta
        

app = Flask(__name__)
CORS(app)

model = joblib.load('demand_model.pkl')



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

        

        df = pd.read_csv('../data/train.csv')
        df['date'] = pd.to_datetime(df['date'])

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

if __name__ == '__main__':
    app.run(debug=True)