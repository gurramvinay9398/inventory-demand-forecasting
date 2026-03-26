# 🚀 AI-Powered Inventory Demand Forecasting System

## 📌 Overview
This project is a full-stack AI-based Inventory Management System that predicts product demand, provides stock recommendations, and offers business insights through an interactive dashboard.

It simulates a real-world retail system (like Blinkit/Amazon) where businesses can optimize inventory using data-driven decisions.

---

## 🎯 Features

### 🔹 Demand Forecasting
- Predicts future demand for products
- Uses machine learning model (Random Forest)

### 🔹 Inventory Optimization
- Suggests recommended stock levels
- Prevents overstock and understock situations

### 🔹 Smart Alerts
- Low stock alert
- Overstock alert

### 🔹 Analytics Dashboard
- 📈 Demand trend visualization
- 🏬 Store-wise performance analysis
- 🔥 Top-selling items (dynamic filtering)
- Interactive charts using Chart.js

### 🔹 Dynamic Filters
- Select number of days for trend analysis
- Choose Top N items and stores
- Real-time updates from backend

### 🔹 Modern UI (Product-Level)
- Landing page (home.html)
- Dashboard (index.html)
- Analytics page (analytics.html)
- Smooth navigation and hover effects

---

## 🛠️ Tech Stack

### 🔹 Frontend
- HTML, CSS, JavaScript
- Chart.js (for visualization)

### 🔹 Backend
- Python
- Flask (REST API)

### 🔹 Machine Learning
- Scikit-learn
- Random Forest Regressor

### 🔹 Tools
- Git & GitHub
- VS Code

---

## 📊 Project Structure
Inventory demand forecasting/
│
├── backend/
│ ├── app.py
│ ├── model.py
| ├──check.py
| ├──predict_future.py
├── frontend/
│ ├── home.html
│ ├── index.html
│ ├── analytics.html
│ ├── image1.img
│ ├── image2.img
│
├── data/
│ ├── train.csv
│
└── README.md


---

## ⚙️ How to Run the Project

### 🔹 Step 1: Clone Repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd Inventory-demand-forecasting

🔹 Step 2: Setup Virtual Environment
python -m venv venv
venv\Scripts\activate   # Windows

🔹 Step 3: Install Dependencies
pip install -r requirements.txt

🔹 Step 4: Run Backend
cd backend
python app.py

Step 5: Open Frontend

Open in browser:
frontend/home.html


📈 API Endpoints
🔹 Prediction API
POST /predict
🔹 Analytics API
POST /analytics

👨‍💻 Author
Vinay Gurram