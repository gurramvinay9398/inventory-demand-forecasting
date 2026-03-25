from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

model = joblib.load('sales_model.pkl')

PRODUCT_COLUMNS = [
    "Product line_Electronic accessories",
    "Product line_Fashion accessories",
    "Product line_Food and beverages",
    "Product line_Health and beauty",
    "Product line_Home and lifestyle",
    "Product line_Sports and travel"
]

@app.route('/')
def home():
    return "Sales Prediction API Running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        input_data = {
            "Unit price": float(data.get("Unit price", 0)),
            "Quantity": int(data.get("Quantity", 0)),
            "day": int(data.get("day", 1)),
            "month": int(data.get("month", 1)),
            "day_of_week": int(data.get("day_of_week", 0))
        }

        # Initialize all product columns = 0
        for col in PRODUCT_COLUMNS:
            input_data[col] = 0

        # Set selected product = 1
        selected_product = data.get("product")
        if selected_product in PRODUCT_COLUMNS:
            input_data[selected_product] = 1

        df = pd.DataFrame([input_data])

        prediction = model.predict(df)[0]

        return jsonify({
            "predicted_sales": round(float(prediction), 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)