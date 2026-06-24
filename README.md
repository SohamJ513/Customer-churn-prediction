---
title: Customer Churn Prediction System
emoji: 🔮
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# 🔮 Customer Churn Prediction System

ML-based web application to predict customer churn for telecom companies.

## 📊 Model Performance

| Model | Accuracy | Precision | Recall |
|-------|----------|-----------|--------|
| Logistic Regression | 82.4% | 0.81 | 0.79 |
| Decision Tree | 79.1% | 0.76 | 0.72 |
| Random Forest | 80.7% | 0.79 | 0.76 |

## 🎯 Features

- Customer churn prediction using multiple ML models
- Interactive dashboard with performance metrics
- What-if analysis for scenario testing
- Automatic customer segmentation
- Historical predictions tracking
- Feature importance visualization

## 🛠️ Technologies

- **Backend**: Flask (Web framework)
- **ML Models**: Scikit-learn (3 models trained from scratch)
- **Data Processing**: Pandas & NumPy
- **Visualizations**: Matplotlib, Seaborn, Plotly
- **Deployment**: Docker, Hugging Face Spaces

## 📁 Project Structure
├── app.py # Flask web application
├── requirements.txt # Python dependencies
├── Dockerfile # Container configuration
├── templates/ # HTML templates
├── models/ # Trained ML models (.pkl files)
├── static/ # CSS and JavaScript files
└── data/ # Dataset files

text

## 🚀 Installation

1. Clone the repository
```bash
git clone <your-repo-url>
cd Customer-churn-prediction-using-ML

Create a virtual environment
bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies
bash
pip install -r requirements.txt
Run the application

bash
python app.py

🤖 Models Used
Logistic Regression - Gradient descent optimized with L2 regularization

Decision Tree - Entropy-based splitting with depth optimization

Random Forest - 30 trees with bootstrap sampling & feature bagging

📊 Dataset
Telco Customer Churn dataset from IBM

🐳 Docker Deployment
Build the Docker image

bash
docker build -t churn-predictor .
Run the container

bash
docker run -p 7860:7860 churn-predictor

📝 License
MIT License - feel free to use and modify!

