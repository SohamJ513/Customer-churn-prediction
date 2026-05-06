# Customer Churn Prediction System

ML-based web application to predict customer churn for telecom companies.

## Features
- Customer churn prediction using multiple ML models
- Interactive dashboard
- What-if analysis
- Customer segmentation
- Historical predictions tracking

## Technologies
- Flask (Web framework)
- Scikit-learn (ML models)
- Pandas & NumPy (Data processing)
- Matplotlib & Seaborn (Visualizations)

## Installation

1. Clone the repository
```bash
git clone <your-repo-url>
cd Customer-churn-prediction-using-ML

2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install dependencies
pip install -r requirements.txt

4. Run the application
python app.py

Models Used
1. Logistic Regression
2. Decision Tree
3. Random Forest

Project Structure
├── app/           # Flask web application
├── src/           # ML model code
├── data/          # Dataset
├── notebooks/     # EDA notebooks
└── models/        # Saved models (.pkl files - not in git)

Dataset 
Telco Customer Churn dataset from IBM

## **Step 3: Initialize Git and push**

Run these commands in your terminal:

```bash
# Initialize git repository
git init

# Add all necessary files
git add .gitignore
git add requirements.txt
git add README.md
git add src/
git add app/
git add notebooks/
git add data/
git add *.py

# Check what will be committed
git status

# Commit the files
git commit -m "Initial commit: Customer churn prediction system"

# Add your GitHub repository (replace with your repo URL)
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Push to GitHub
git branch -M main
git push -u origin main