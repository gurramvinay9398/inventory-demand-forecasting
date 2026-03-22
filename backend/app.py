from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model
model = joblib.load('sales_model.pkl')

@app.route('/')
def home():
    return "Sales Prediction API Running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Convert input into DataFrame
    df = pd.DataFrame([data])

    prediction = model.predict(df)

    return jsonify({
        'predicted_sales': prediction[0]
    })

if __name__ == '__main__':
    app.run(debug=True)