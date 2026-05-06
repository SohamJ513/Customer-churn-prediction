from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import os
import sys
import json
import plotly
import plotly.graph_objs as go
from collections import deque
from datetime import datetime
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Add parent directory to path to import your modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your model classes
from src.logistic_regression import LogisticRegressionFromScratch
from src.decision_tree import DecisionTreeFromScratch
from src.random_forest import RandomForestFromScratch

app = Flask(__name__)

# Get the root directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT_DIR, 'models')

print(f"📁 Root directory: {ROOT_DIR}")
print(f"📁 Models directory: {MODELS_DIR}")

# File to store prediction history
HISTORY_FILE = os.path.join(ROOT_DIR, 'prediction_history.json')

def load_history():
    """Load prediction history from file"""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                data = json.load(f)
                # Convert list back to deque
                print(f"✅ Loaded {len(data)} predictions from history file")
                return deque(data, maxlen=50)
        except Exception as e:
            print(f"⚠️ Could not load history file: {e}")
            return deque(maxlen=50)
    else:
        print("📁 No history file found, starting fresh")
        return deque(maxlen=50)

def save_history(history):
    """Save prediction history to file"""
    try:
        with open(HISTORY_FILE, 'w') as f:
            # Convert deque to list for JSON serialization
            json.dump(list(history), f, indent=2)
        print(f"💾 Saved {len(history)} predictions to history file")
    except Exception as e:
        print(f"⚠️ Could not save history: {e}")

# Load history at startup
recent_predictions = load_history()

# Helper function to format features for display
def format_features_for_display(form_values):
    """Convert form values to readable format"""
    features_display = []
    
    # Map of feature names to readable labels
    feature_labels = {
        'gender': 'Male',
        'SeniorCitizen': 'Senior Citizen',
        'Partner': 'Has Partner',
        'Dependents': 'Has Dependents',
        'PhoneService': 'Phone Service',
        'MultipleLines_Yes': 'Multiple Lines',
        'InternetService_Fiber optic': 'Fiber Optic Internet',
        'InternetService_No': 'No Internet',
        'OnlineSecurity_Yes': 'Online Security',
        'OnlineBackup_Yes': 'Online Backup',
        'DeviceProtection_Yes': 'Device Protection',
        'TechSupport_Yes': 'Tech Support',
        'StreamingTV_Yes': 'Streaming TV',
        'StreamingMovies_Yes': 'Streaming Movies',
        'Contract_One year': 'One Year Contract',
        'Contract_Two year': 'Two Year Contract',
        'PaperlessBilling': 'Paperless Billing',
        'PaymentMethod_Credit card (automatic)': 'Credit Card (Auto)',
        'PaymentMethod_Electronic check': 'Electronic Check',
        'PaymentMethod_Mailed check': 'Mailed Check'
    }
    
    # Add enabled features to display
    for key, value in form_values.items():
        if value == 1.0 and key in feature_labels:
            features_display.append(feature_labels[key])
    
    # Add tenure separately
    tenure = form_values.get('tenure', 0)
    if tenure:
        features_display.append(f'Tenure: {int(tenure)} months')
    
    return features_display[:5]  # Return top 5 features for display

# Feature labels for display in detail view
feature_labels_display = {
    'gender': 'Gender (Male)',
    'SeniorCitizen': 'Senior Citizen',
    'Partner': 'Has Partner',
    'Dependents': 'Has Dependents',
    'tenure': 'Tenure',
    'PhoneService': 'Phone Service',
    'MultipleLines_Yes': 'Multiple Lines',
    'InternetService_Fiber optic': 'Fiber Optic Internet',
    'InternetService_No': 'No Internet',
    'OnlineSecurity_Yes': 'Online Security',
    'OnlineBackup_Yes': 'Online Backup',
    'DeviceProtection_Yes': 'Device Protection',
    'TechSupport_Yes': 'Tech Support',
    'StreamingTV_Yes': 'Streaming TV',
    'StreamingMovies_Yes': 'Streaming Movies',
    'Contract_One year': 'One Year Contract',
    'Contract_Two year': 'Two Year Contract',
    'PaperlessBilling': 'Paperless Billing',
    'PaymentMethod_Credit card (automatic)': 'Credit Card (Auto)',
    'PaymentMethod_Electronic check': 'Electronic Check',
    'PaymentMethod_Mailed check': 'Mailed Check',
    'MonthlyCharges': 'Monthly Charges',
    'TotalCharges': 'Total Charges'
}

# Load feature names
try:
    with open(os.path.join(MODELS_DIR, 'feature_names.pkl'), 'rb') as f:
        FEATURE_NAMES = pickle.load(f)
    print(f"✅ Loaded {len(FEATURE_NAMES)} feature names")
    print(f"   First 10: {FEATURE_NAMES[:10]}")
except Exception as e:
    print(f"⚠️ Could not load feature names: {e}")
    FEATURE_NAMES = None

# Model info for display
model_info = {
    'logistic': {
        'name': 'Logistic Regression',
        'accuracy': 82.39,
        'color': '#3498db',
        'description': 'Linear model with gradient descent'
    },
    'tree': {
        'name': 'Decision Tree',
        'accuracy': 79.12,
        'color': '#f39c12',
        'description': 'Tree-based model with entropy splitting'
    },
    'forest': {
        'name': 'Random Forest',
        'accuracy': 80.68,
        'color': '#2ecc71',
        'description': 'Ensemble of 30 trees with bootstrap sampling'
    }
}

# Load models
models = {}

# Load Logistic Regression (special handling because it's saved as dict)
logistic_path = os.path.join(MODELS_DIR, 'optimized_model.pkl')
try:
    if os.path.exists(logistic_path):
        with open(logistic_path, 'rb') as f:
            logistic_dict = pickle.load(f)
        
        # Reconstruct Logistic Regression model
        logistic_model = LogisticRegressionFromScratch()
        logistic_model.weights = logistic_dict['weights']
        logistic_model.bias = logistic_dict['bias']
        logistic_model.feature_names = FEATURE_NAMES
        models['logistic'] = logistic_model
        print(f"✅ Logistic Regression loaded and reconstructed")
    else:
        print(f"❌ Logistic Regression not found at {logistic_path}")
        models['logistic'] = None
except Exception as e:
    print(f"❌ Error loading Logistic Regression: {e}")
    models['logistic'] = None

# Load Decision Tree
tree_path = os.path.join(MODELS_DIR, 'decision_tree_best.pkl')
try:
    if os.path.exists(tree_path):
        with open(tree_path, 'rb') as f:
            models['tree'] = pickle.load(f)
        print(f"✅ Decision Tree loaded from {tree_path}")
    else:
        print(f"❌ Decision Tree not found at {tree_path}")
        models['tree'] = None
except Exception as e:
    print(f"❌ Error loading Decision Tree: {e}")
    models['tree'] = None

# Load Random Forest
forest_path = os.path.join(MODELS_DIR, 'random_forest_best.pkl')
try:
    if os.path.exists(forest_path):
        with open(forest_path, 'rb') as f:
            models['forest'] = pickle.load(f)
        print(f"✅ Random Forest loaded from {forest_path}")
    else:
        print(f"❌ Random Forest not found at {forest_path}")
        models['forest'] = None
except Exception as e:
    print(f"❌ Error loading Random Forest: {e}")
    models['forest'] = None

# Feature info for the form (simplified for UI)
feature_info = [
    {'name': 'gender', 'label': 'Gender (Male)', 'type': 'checkbox'},
    {'name': 'SeniorCitizen', 'label': 'Senior Citizen', 'type': 'checkbox'},
    {'name': 'Partner', 'label': 'Has Partner', 'type': 'checkbox'},
    {'name': 'Dependents', 'label': 'Has Dependents', 'type': 'checkbox'},
    {'name': 'tenure', 'label': 'Tenure (months)', 'type': 'number', 'default': 12, 'min': 0, 'max': 100},
    {'name': 'PhoneService', 'label': 'Has Phone Service', 'type': 'checkbox'},
    {'name': 'MultipleLines_Yes', 'label': 'Has Multiple Lines', 'type': 'checkbox'},
    {'name': 'InternetService_Fiber optic', 'label': 'Fiber Optic Internet', 'type': 'checkbox'},
    {'name': 'InternetService_No', 'label': 'No Internet Service', 'type': 'checkbox'},
    {'name': 'OnlineSecurity_Yes', 'label': 'Has Online Security', 'type': 'checkbox'},
    {'name': 'OnlineBackup_Yes', 'label': 'Has Online Backup', 'type': 'checkbox'},
    {'name': 'DeviceProtection_Yes', 'label': 'Has Device Protection', 'type': 'checkbox'},
    {'name': 'TechSupport_Yes', 'label': 'Has Tech Support', 'type': 'checkbox'},
    {'name': 'StreamingTV_Yes', 'label': 'Has Streaming TV', 'type': 'checkbox'},
    {'name': 'StreamingMovies_Yes', 'label': 'Has Streaming Movies', 'type': 'checkbox'},
    {'name': 'Contract_One year', 'label': 'One Year Contract', 'type': 'checkbox'},
    {'name': 'Contract_Two year', 'label': 'Two Year Contract', 'type': 'checkbox'},
    {'name': 'PaperlessBilling', 'label': 'Paperless Billing', 'type': 'checkbox'},
    {'name': 'PaymentMethod_Credit card (automatic)', 'label': 'Credit Card (Auto)', 'type': 'checkbox'},
    {'name': 'PaymentMethod_Electronic check', 'label': 'Electronic Check', 'type': 'checkbox'},
    {'name': 'PaymentMethod_Mailed check', 'label': 'Mailed Check', 'type': 'checkbox'}
]

# ===== LANDING PAGE ROUTE =====
@app.route('/')
def landing():
    """Landing page"""
    return render_template('landing.html')

# ===== APP ROUTES =====
@app.route('/app')
def app_index():
    """Main application page"""
    return render_template('index.html', feature_info=feature_info, models=model_info)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        model_key = request.form.get('model', 'logistic')
        
        # Create array with ALL 30 features (initialize with zeros)
        features = np.zeros(len(FEATURE_NAMES))
        
        # First pass: collect all form inputs
        form_values = {}
        for feature in feature_info:
            if feature['type'] == 'checkbox':
                form_values[feature['name']] = 1.0 if request.form.get(feature['name']) == 'on' else 0.0
            elif feature['name'] == 'tenure':
                try:
                    form_values['tenure'] = float(request.form.get('tenure', 12))
                except:
                    form_values['tenure'] = 12.0
        
        # Map form inputs to features
        for i, feature_name in enumerate(FEATURE_NAMES):
            if feature_name in form_values:
                features[i] = form_values[feature_name]
            elif feature_name == 'tenure':
                features[i] = form_values.get('tenure', 12.0)
        
        # ===== MORE ACCURATE MonthlyCharges and TotalCharges CALCULATION =====
        try:
            monthly_idx = FEATURE_NAMES.index('MonthlyCharges')
            total_idx = FEATURE_NAMES.index('TotalCharges')
            
            # ===== MORE ACCURATE MonthlyCharges CALCULATION =====
            monthly_base = 20.0  # Base charge
            
            # Phone service
            if form_values.get('PhoneService', 0) == 1:
                monthly_base += 20.0  # Phone service cost
            
            # Multiple lines (only if they have phone service)
            if form_values.get('PhoneService', 0) == 1 and form_values.get('MultipleLines_Yes', 0) == 1:
                monthly_base += 10.0  # Additional for multiple lines
            
            # Internet service - more realistic pricing
            if form_values.get('InternetService_Fiber optic', 0) == 1:
                monthly_base += 70.0  # Fiber optic (premium)
            elif form_values.get('InternetService_No', 0) == 1:
                monthly_base += 0  # No internet
            else:
                monthly_base += 50.0  # DSL (standard)
            
            # Online services - only if they have internet
            if form_values.get('InternetService_No', 0) == 0:  # Has internet
                # Each online service adds $5-10, not $15
                if form_values.get('OnlineSecurity_Yes', 0) == 1:
                    monthly_base += 8.0
                if form_values.get('OnlineBackup_Yes', 0) == 1:
                    monthly_base += 8.0
                if form_values.get('DeviceProtection_Yes', 0) == 1:
                    monthly_base += 8.0
                if form_values.get('TechSupport_Yes', 0) == 1:
                    monthly_base += 15.0  # Tech support is premium
                if form_values.get('StreamingTV_Yes', 0) == 1:
                    monthly_base += 12.0
                if form_values.get('StreamingMovies_Yes', 0) == 1:
                    monthly_base += 12.0
            
            # Cap monthly charges at a realistic maximum
            monthly_base = min(monthly_base, 140.0)  # Most customers don't pay >$140
            
            # Get tenure
            tenure = form_values.get('tenure', 12)
            
            # Set the values
            features[monthly_idx] = monthly_base
            features[total_idx] = monthly_base * tenure
            
            print(f"\n💰 BEFORE SCALING:")
            print(f"   Monthly Charges: ${monthly_base:.2f}")
            print(f"   Total Charges: ${monthly_base * tenure:.2f}")
            print(f"   Tenure: {tenure} months")
            
        except ValueError as e:
            print(f"⚠️ Could not find MonthlyCharges/TotalCharges indices: {e}")
        
        # ===== COMPLETE FEATURE MAPPING =====
        print("\n🔄 Applying complete feature mapping...")
        
        # Get internet service type
        has_internet = form_values.get('InternetService_No', 0) == 0
        is_fiber = form_values.get('InternetService_Fiber optic', 0) == 1
        is_dsl = has_internet and not is_fiber and form_values.get('InternetService_No', 0) == 0
        
        # Set all the "No internet service" flags correctly
        for i, feature_name in enumerate(FEATURE_NAMES):
            if 'No internet service' in feature_name:
                # These should be 1 ONLY if customer has NO internet
                features[i] = 1.0 if not has_internet else 0.0
            
            # Also handle specific feature mappings
            elif feature_name == 'MultipleLines_No phone service':
                features[i] = 1.0 if form_values.get('PhoneService', 0) == 0 else 0.0
            
            elif feature_name == 'OnlineSecurity_No internet service':
                features[i] = 1.0 if not has_internet else 0.0
            
            elif feature_name == 'OnlineBackup_No internet service':
                features[i] = 1.0 if not has_internet else 0.0
            
            elif feature_name == 'DeviceProtection_No internet service':
                features[i] = 1.0 if not has_internet else 0.0
            
            elif feature_name == 'TechSupport_No internet service':
                features[i] = 1.0 if not has_internet else 0.0
            
            elif feature_name == 'StreamingTV_No internet service':
                features[i] = 1.0 if not has_internet else 0.0
            
            elif feature_name == 'StreamingMovies_No internet service':
                features[i] = 1.0 if not has_internet else 0.0
        
        print("✅ Feature mapping complete")
        
        # ===== ADD DEBUG INFO FOR LOW RISK CASE =====
        print("\n" + "="*50)
        print("🔍 DEBUG INFO - Raw Values Before Scaling")
        print("="*50)
        
        # Print key raw values
        key_features_raw = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract_Two year', 
                           'PaymentMethod_Credit card (automatic)', 'InternetService_Fiber optic']
        for feature_name in key_features_raw:
            if feature_name in FEATURE_NAMES:
                idx = FEATURE_NAMES.index(feature_name)
                print(f"{feature_name:30} = {features[idx]:.4f}")
        
        # ===== CRITICAL: SCALE NUMERICAL FEATURES to 0-1 RANGE =====
        print(f"\n📊 APPLYING SCALING (0-1 range):")
        
        # Scale tenure (max 72 months)
        if 'tenure' in FEATURE_NAMES:
            idx = FEATURE_NAMES.index('tenure')
            original_tenure = features[idx]
            features[idx] = features[idx] / 72.0  # Max tenure is 72
            print(f"   tenure: {original_tenure:.1f} → {features[idx]:.4f}")
        
        # Scale MonthlyCharges (max 140 now)
        if 'MonthlyCharges' in FEATURE_NAMES:
            idx = FEATURE_NAMES.index('MonthlyCharges')
            original_monthly = features[idx]
            features[idx] = features[idx] / 140.0  # Updated max to 140
            print(f"   MonthlyCharges: ${original_monthly:.2f} → {features[idx]:.4f}")
        
        # Scale TotalCharges (max 140*72 ≈ 10000)
        if 'TotalCharges' in FEATURE_NAMES:
            idx = FEATURE_NAMES.index('TotalCharges')
            original_total = features[idx]
            features[idx] = features[idx] / 10000.0  # Updated max
            print(f"   TotalCharges: ${original_total:.2f} → {features[idx]:.4f}")
        
        # ===== FINAL FEATURE VERIFICATION =====
        print("\n" + "="*60)
        print("🔍 FINAL FEATURE VALUES (POST-SCALING)")
        print("="*60)
        
        # Print key features after scaling
        key_features_scaled = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract_Two year',
                              'PaymentMethod_Credit card (automatic)', 'OnlineSecurity_Yes']
        
        for feature_name in key_features_scaled:
            if feature_name in FEATURE_NAMES:
                idx = FEATURE_NAMES.index(feature_name)
                print(f"{feature_name:30} = {features[idx]:.4f}")
        
        # Check if any "No internet service" flags are incorrectly set
        print("\n📌 Checking 'No internet service' flags:")
        no_internet_flags_found = False
        for feature_name in FEATURE_NAMES:
            if 'No internet service' in feature_name:
                idx = FEATURE_NAMES.index(feature_name)
                if features[idx] == 1.0:
                    print(f"   ⚠️  {feature_name} = 1.0")
                    no_internet_flags_found = True
        
        if not no_internet_flags_found:
            print("   ✅ No 'No internet service' flags are set (correct for customers with internet)")
        
        X = features.reshape(1, -1)
        print(f"\n📊 Final feature vector shape: {X.shape}")
        
        # Get the model
        model = models.get(model_key)
        
        if model is None:
            return render_template('result.html', result={'error': f'Model {model_key} not available'})
        
        # Make prediction
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)[0]
            # Handle different return types
            if isinstance(proba, np.ndarray) and len(proba) >= 2:
                churn_prob = float(proba[1] * 100)
                confidence = float(max(proba) * 100)
                prediction = 1 if proba[1] > 0.5 else 0
                print(f"\n✅ Model prediction: {'🔴 CHURN' if prediction == 1 else '🟢 STAY'}")
                print(f"   Confidence: {confidence:.2f}%")
                print(f"   Churn Probability: {churn_prob:.2f}%")
            else:
                # Fallback if proba format is unexpected
                prediction = int(model.predict(X)[0])
                churn_prob = 75.0 if prediction == 1 else 25.0
                confidence = 75.0
                print(f"⚠️ Model predict_proba returned unexpected format")
        else:
            prediction = int(model.predict(X)[0])
            churn_prob = 75.0 if prediction == 1 else 25.0
            confidence = 75.0
            print(f"⚠️ Model has no predict_proba method")
        
        # Get predictions from all models for comparison
        all_predictions = {}
        for key, m in models.items():
            if m is not None:
                try:
                    if hasattr(m, 'predict_proba'):
                        p = m.predict_proba(X)[0]
                        if isinstance(p, np.ndarray) and len(p) >= 2:
                            pred = 'CHURN' if p[1] > 0.5 else 'STAY'
                            conf = float(max(p) * 100)
                            prob = float(p[1] * 100)
                        else:
                            pred_val = int(m.predict(X)[0])
                            pred = 'CHURN' if pred_val == 1 else 'STAY'
                            conf = 75.0
                            prob = 75.0 if pred_val == 1 else 25.0
                    else:
                        pred_val = int(m.predict(X)[0])
                        pred = 'CHURN' if pred_val == 1 else 'STAY'
                        conf = 75.0
                        prob = 75.0 if pred_val == 1 else 25.0
                except Exception as e:
                    print(f"⚠️ Error predicting with {key}: {e}")
                    pred = 'STAY'
                    conf = 50.0
                    prob = 50.0
                
                all_predictions[model_info[key]['name']] = {
                    'prediction': pred,
                    'confidence': round(conf, 2),
                    'churn_prob': round(prob, 2)
                }
        
        result = {
            'success': True,
            'model_used': model_info[model_key]['name'],
            'prediction': 'CHURN' if prediction == 1 else 'STAY',
            'confidence': round(confidence, 2),
            'churn_probability': round(churn_prob, 2),
            'all_predictions': all_predictions
        }
        
        # ===== STORE PREDICTION IN HISTORY =====
        try:
            # Get the model key that was used
            model_key = request.form.get('model', 'logistic')
            
            # Create a readable prediction entry
            prediction_entry = {
                'id': len(recent_predictions) + 1,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'model': model_info[model_key]['name'],
                'model_key': model_key,
                'prediction': result['prediction'],
                'confidence': result['confidence'],
                'churn_prob': result['churn_probability'],
                'features': format_features_for_display(form_values),
                'full_features': dict(form_values)
            }
            
            recent_predictions.appendleft(prediction_entry)
            save_history(recent_predictions)  # Save to file
            print(f"✅ Prediction stored in history. Total: {len(recent_predictions)}")
        except Exception as e:
            print(f"⚠️ Could not store prediction in history: {e}")
        
        return render_template('result.html', result=result)
    
    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        import traceback
        traceback.print_exc()
        return render_template('result.html', result={'error': str(e)})

# ===== WHAT-IF ANALYSIS ROUTES =====

@app.route('/what-if')
def what_if():
    """Interactive what-if analysis page"""
    return render_template('what_if.html', 
                         feature_info=feature_info, 
                         models=model_info,
                         feature_labels=feature_labels_display)

@app.route('/api/what-if', methods=['POST'])
def what_if_api():
    """API for real-time predictions"""
    try:
        data = request.get_json()
        
        # Extract features from the request
        features = data['features']
        model_key = data.get('model', 'forest')
        
        # DEBUG: Print the features
        print(f"\n🔍 WHAT-IF API CALLED")
        print(f"   Model: {model_key}")
        print(f"   Features length: {len(features)}")
        
        # Convert to numpy array and reshape
        X = np.array(features, dtype=np.float64).reshape(1, -1)
        print(f"   X shape: {X.shape}")
        print(f"   X dtype: {X.dtype}")
        
        model = models.get(model_key)
        
        if model is None:
            return jsonify({'error': 'Model not found'}), 404
        
        # Make prediction for the selected model
        churn_prob = 25.0
        prediction = 'STAY'
        confidence = 75.0
        
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)[0]
            print(f"   Probabilities for {model_key}: {proba}")
            
            # Handle different return types
            if isinstance(proba, (list, tuple, np.ndarray)) and len(proba) >= 2:
                churn_prob = float(proba[1] * 100)
                prediction = 'CHURN' if proba[1] > 0.5 else 'STAY'
                confidence = float(max(proba) * 100)
            elif isinstance(proba, (int, float)):
                # If it's a single number, assume it's the probability of class 1
                churn_prob = float(proba * 100)
                prediction = 'CHURN' if proba > 0.5 else 'STAY'
                confidence = max(churn_prob, 100 - churn_prob)
                print(f"   Single value detected: {proba}")
            else:
                print(f"   Unexpected proba format: {type(proba)}")
        else:
            pred = int(model.predict(X)[0])
            churn_prob = 75.0 if pred == 1 else 25.0
            prediction = 'CHURN' if pred == 1 else 'STAY'
            confidence = 75.0
        
        # Get predictions from all models for comparison
        all_predictions = {}
        for key, m in models.items():
            if m is not None:
                try:
                    if hasattr(m, 'predict_proba'):
                        p = m.predict_proba(X)[0]
                        
                        # Handle different return types for each model
                        if isinstance(p, (list, tuple, np.ndarray)) and len(p) >= 2:
                            pred_val = 'CHURN' if p[1] > 0.5 else 'STAY'
                            conf_val = float(max(p) * 100)
                            prob_val = float(p[1] * 100)
                        elif isinstance(p, (int, float)):
                            # Single value case
                            prob_val = float(p * 100)
                            pred_val = 'CHURN' if p > 0.5 else 'STAY'
                            conf_val = max(prob_val, 100 - prob_val)
                        else:
                            pred_val = 'STAY'
                            conf_val = 50.0
                            prob_val = 25.0
                            print(f"   Unexpected format for {key}: {type(p)}")
                    else:
                        pred_val = int(m.predict(X)[0])
                        pred_val = 'CHURN' if pred_val == 1 else 'STAY'
                        conf_val = 75.0
                        prob_val = 75.0 if pred_val == 'CHURN' else 25.0
                    
                    all_predictions[model_info[key]['name']] = {
                        'prediction': pred_val,
                        'confidence': round(conf_val, 2),
                        'churn_prob': round(prob_val, 2)
                    }
                except Exception as e:
                    print(f"   Error predicting with {key}: {e}")
                    all_predictions[model_info[key]['name']] = {
                        'prediction': 'STAY',
                        'confidence': 50.0,
                        'churn_prob': 25.0
                    }
        
        return jsonify({
            'success': True,
            'churn_probability': round(churn_prob, 2),
            'prediction': prediction,
            'confidence': round(confidence, 2),
            'all_predictions': all_predictions
        })
    
    except Exception as e:
        print(f"❌ Error in what-if API: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ===== HISTORY ROUTES =====

@app.route('/history')
def history():
    """Show recent predictions history"""
    return render_template('history.html', 
                         predictions=list(recent_predictions),
                         model_info=model_info)

@app.route('/history/<int:prediction_id>')
def history_detail(prediction_id):
    """Show detailed view of a specific prediction"""
    try:
        # Find the prediction by ID
        prediction = None
        for p in recent_predictions:
            if p['id'] == prediction_id:
                prediction = p
                break
        
        if prediction is None:
            return render_template('error.html', error=f'Prediction {prediction_id} not found')
        
        # Get feature importance for the model used
        model_key = prediction.get('model_key', 'logistic')
        model = models.get(model_key)
        
        feature_importance = {}
        if model is not None:
            if model_key == 'logistic' and hasattr(model, 'weights'):
                weights = np.abs(model.weights)
                for i, name in enumerate(FEATURE_NAMES[:10]):
                    if i < len(weights):
                        feature_importance[name] = float(weights[i])
            elif hasattr(model, 'get_feature_importance'):
                imp = model.get_feature_importance()
                sorted_imp = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:10]
                feature_importance = dict(sorted_imp)
        
        return render_template('history_detail.html', 
                             prediction=prediction,
                             feature_importance=feature_importance,
                             model_info=model_info,
                             feature_labels=feature_labels_display)
    
    except Exception as e:
        print(f"Error in history_detail: {e}")
        return render_template('error.html', error=str(e))

@app.route('/history/clear', methods=['POST'])
def clear_history():
    """Clear prediction history"""
    recent_predictions.clear()
    save_history(recent_predictions)  # Save empty history to file
    return jsonify({'success': True, 'message': 'History cleared'})

@app.route('/history/export')
def export_history():
    """Export history as JSON"""
    history_list = list(recent_predictions)
    return jsonify(history_list)

# ===== DASHBOARD ROUTE =====

@app.route('/dashboard')
def dashboard():
    """Show model performance dashboard"""
    try:
        # Model performance metrics
        metrics = {
            'logistic': {
                'name': 'Logistic Regression',
                'accuracy': 82.39,
                'precision': 0.78,
                'recall': 0.65,
                'f1': 0.71,
                'color': '#3498db'
            },
            'tree': {
                'name': 'Decision Tree',
                'accuracy': 79.12,
                'precision': 0.75,
                'recall': 0.62,
                'f1': 0.68,
                'color': '#f39c12'
            },
            'forest': {
                'name': 'Random Forest',
                'accuracy': 80.68,
                'precision': 0.77,
                'recall': 0.64,
                'f1': 0.70,
                'color': '#2ecc71'
            }
        }
        
        # Get feature importance for each model for the charts
        feature_importance_data = {}
        for model_name, model in models.items():
            if model is not None:
                importance_data = {}
                if model_name == 'logistic' and hasattr(model, 'weights'):
                    weights = np.abs(model.weights)
                    # Get top 10 features
                    indices = np.argsort(weights)[-10:][::-1]
                    for i in indices[:10]:
                        if i < len(FEATURE_NAMES):
                            importance_data[FEATURE_NAMES[i]] = float(weights[i])
                elif hasattr(model, 'get_feature_importance'):
                    imp = model.get_feature_importance()
                    # Get top 5
                    sorted_imp = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:5]
                    importance_data = dict(sorted_imp)
                elif model_name == 'forest' and hasattr(model, 'feature_importances_'):
                    imp = model.feature_importances_
                    # Get top 5
                    if isinstance(imp, dict):
                        sorted_imp = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:5]
                        importance_data = dict(sorted_imp)
                
                feature_importance_data[model_name] = importance_data
        
        return render_template('dashboard.html', 
                             metrics=metrics,
                             feature_importance=feature_importance_data)
    
    except Exception as e:
        print(f"Error in dashboard: {e}")
        return render_template('dashboard.html', error=str(e))

# ===== FEATURE IMPORTANCE ROUTES =====

@app.route('/feature_importance/<model_name>')
def feature_importance(model_name):
    """Get feature importance for a specific model"""
    try:
        model = models.get(model_name)
        if model is None:
            return jsonify({'error': 'Model not found'}), 404
        
        importance_data = {}
        
        # Get feature importance based on model type
        if model_name == 'logistic' and hasattr(model, 'weights'):
            # For logistic regression, use absolute weights
            weights = np.abs(model.weights)
            for i, name in enumerate(FEATURE_NAMES):
                if i < len(weights):
                    importance_data[name] = float(weights[i])
        
        elif model_name == 'tree' and hasattr(model, 'get_feature_importance'):
            # For decision tree
            importance_data = model.get_feature_importance()
        
        elif model_name == 'forest' and hasattr(model, 'feature_importances_'):
            # For random forest
            importance_data = model.feature_importances_
        
        # Sort and get top 15 features
        sorted_importance = sorted(importance_data.items(), key=lambda x: x[1], reverse=True)[:15]
        
        # Create plotly chart
        features = [item[0][:25] + '...' if len(item[0]) > 25 else item[0] for item in sorted_importance]
        importance = [item[1] for item in sorted_importance]
        
        # Create horizontal bar chart
        fig = go.Figure(data=go.Bar(
            x=importance,
            y=features,
            orientation='h',
            marker=dict(
                color=importance,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Importance")
            ),
            text=[f'{imp:.3f}' for imp in importance],
            textposition='outside',
        ))
        
        fig.update_layout(
            title=f'Top 15 Feature Importance - {model_info[model_name]["name"]}',
            xaxis_title='Importance',
            yaxis_title='Features',
            height=600,
            margin=dict(l=200, r=50, t=100, b=50),
            template='plotly_white'
        )
        
        # Convert to JSON for rendering
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        return render_template('feature_importance.html', 
                             graphJSON=graphJSON,
                             model_name=model_name,
                             model_info=model_info[model_name])
    
    except Exception as e:
        print(f"Error in feature_importance: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/feature_importance/<model_name>')
def api_feature_importance(model_name):
    """API endpoint for feature importance data"""
    try:
        model = models.get(model_name)
        if model is None:
            return jsonify({'error': 'Model not found'}), 404
        
        importance_data = {}
        
        if model_name == 'logistic' and hasattr(model, 'weights'):
            weights = np.abs(model.weights)
            for i, name in enumerate(FEATURE_NAMES):
                if i < len(weights):
                    importance_data[name] = float(weights[i])
        elif model_name == 'tree' and hasattr(model, 'get_feature_importance'):
            importance_data = model.get_feature_importance()
        elif model_name == 'forest' and hasattr(model, 'feature_importances_'):
            importance_data = model.feature_importances_
        
        # Sort and get top 15
        sorted_importance = sorted(importance_data.items(), key=lambda x: x[1], reverse=True)[:15]
        
        return jsonify({
            'features': [item[0] for item in sorted_importance],
            'importance': [item[1] for item in sorted_importance]
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'})
        
        model_key = data.get('model', 'logistic')
        features = data.get('features', [])
        
        if not features:
            return jsonify({'error': 'No features provided'})
        
        X = np.array(features).reshape(1, -1)
        model = models.get(model_key)
        
        if model is None:
            return jsonify({'error': 'Model not found'})
        
        result = {
            'model': model_key,
            'prediction': None,
            'confidence': None,
            'churn_probability': None
        }
        
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)[0]
            result['prediction'] = int(proba[1] > 0.5)
            result['confidence'] = float(max(proba))
            result['churn_probability'] = float(proba[1])
        else:
            result['prediction'] = int(model.predict(X)[0])
            result['confidence'] = 0.75
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': {
            'logistic': models.get('logistic') is not None,
            'tree': models.get('tree') is not None,
            'forest': models.get('forest') is not None
        },
        'feature_count': len(FEATURE_NAMES) if FEATURE_NAMES else 0,
        'history_count': len(recent_predictions)
    })

# ===== CUSTOMER SEGMENTATION ROUTE =====

@app.route('/segmentation')
def segmentation():
    """Customer segmentation dashboard"""
    try:
        # Get all predictions from history to analyze
        predictions = list(recent_predictions)
        
        if len(predictions) < 5:
            return render_template('segmentation.html', 
                                 error="Need at least 5 predictions for segmentation. Keep making predictions!")
        
        # Prepare data for clustering
        customer_data = []
        
        for pred in predictions:
            features = pred.get('full_features', {})
            # Extract key numerical features
            monthly = calculate_monthly_from_features(features)
            customer_data.append({
                'tenure': features.get('tenure', 0),
                'monthly_charges': monthly,
                'num_services': count_services(features),
                'has_contract': 1 if features.get('Contract_One year') or features.get('Contract_Two year') else 0,
                'payment_risk': 1 if features.get('PaymentMethod_Electronic check') else 0,
                'churn_prediction': 1 if pred.get('prediction') == 'CHURN' else 0
            })
        
        # Convert to DataFrame
        df = pd.DataFrame(customer_data)
        
        # Perform clustering
        segments, centers, labels = segment_customers(df)
        
        # Create visualization
        plot_url = create_segmentation_plot(df, labels)
        
        # Calculate stats
        high_risk = len([s for s in segments if s['risk'] == 'High'])
        medium_risk = len([s for s in segments if s['risk'] == 'Medium'])
        low_risk = len([s for s in segments if s['risk'] == 'Low'])
        
        return render_template('segmentation.html',
                             segments=segments,
                             plot_url=plot_url,
                             total_customers=len(predictions),
                             stats={
                                 'high_risk': high_risk,
                                 'medium_risk': medium_risk,
                                 'low_risk': low_risk
                             })
    
    except Exception as e:
        print(f"Error in segmentation: {e}")
        import traceback
        traceback.print_exc()
        return render_template('segmentation.html', error=str(e))

def calculate_monthly_from_features(features):
    """Calculate monthly charges from features"""
    monthly = 20.0  # Base
    
    if features.get('PhoneService'):
        monthly += 20.0
        if features.get('MultipleLines_Yes'):
            monthly += 10.0
    
    if features.get('InternetService_Fiber optic'):
        monthly += 70.0
    elif not features.get('InternetService_No'):
        monthly += 50.0
    
    if not features.get('InternetService_No'):
        if features.get('OnlineSecurity_Yes'): monthly += 8.0
        if features.get('OnlineBackup_Yes'): monthly += 8.0
        if features.get('DeviceProtection_Yes'): monthly += 8.0
        if features.get('TechSupport_Yes'): monthly += 15.0
        if features.get('StreamingTV_Yes'): monthly += 12.0
        if features.get('StreamingMovies_Yes'): monthly += 12.0
    
    return min(monthly, 140.0)

def count_services(features):
    """Count number of services customer has"""
    services = 0
    service_list = ['PhoneService', 'MultipleLines_Yes', 'InternetService_Fiber optic',
                   'OnlineSecurity_Yes', 'OnlineBackup_Yes', 'DeviceProtection_Yes',
                   'TechSupport_Yes', 'StreamingTV_Yes', 'StreamingMovies_Yes']
    
    for service in service_list:
        if features.get(service):
            services += 1
    
    return services

def segment_customers(df):
    """Perform K-means clustering to segment customers"""
    # Select features for clustering
    features_for_clustering = df[['tenure', 'monthly_charges', 'num_services', 
                                   'has_contract', 'payment_risk', 'churn_prediction']].values
    
    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_for_clustering)
    
    # Perform K-means clustering (3 segments)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(scaled_features)
    
    # Add cluster labels to dataframe
    df['segment'] = labels
    
    # Analyze each segment
    segments = []
    for i in range(3):
        segment_data = df[df['segment'] == i]
        
        if len(segment_data) == 0:
            continue
        
        # Calculate average metrics
        avg_tenure = segment_data['tenure'].mean()
        avg_monthly = segment_data['monthly_charges'].mean()
        churn_rate = segment_data['churn_prediction'].mean() * 100
        contract_rate = segment_data['has_contract'].mean() * 100
        payment_risk_rate = segment_data['payment_risk'].mean() * 100
        avg_services = segment_data['num_services'].mean()
        
        # Determine risk level
        if churn_rate > 60:
            risk = "High"
            color = "#dc3545"
            action = "Immediate retention call with special offer"
            offer = "20% discount for 6 months + free tech support"
        elif churn_rate > 30:
            risk = "Medium"
            color = "#ffc107"
            action = "Targeted email campaign with loyalty rewards"
            offer = "Free streaming service for 3 months"
        else:
            risk = "Low"
            color = "#28a745"
            action = "Monitor and reward occasionally"
            offer = "Early access to new features"
        
        segments.append({
            'id': i,
            'size': len(segment_data),
            'avg_tenure': round(avg_tenure, 1),
            'avg_monthly': round(avg_monthly, 2),
            'churn_rate': round(churn_rate, 1),
            'contract_rate': round(contract_rate, 1),
            'payment_risk_rate': round(payment_risk_rate, 1),
            'avg_services': round(avg_services, 1),
            'risk': risk,
            'color': color,
            'action': action,
            'offer': offer
        })
    
    return segments, kmeans.cluster_centers_, labels

def create_segmentation_plot(df, labels):
    """Create a 2D visualization of customer segments"""
    plt.figure(figsize=(10, 6))
    
    # Define colors for segments
    colors = ['#28a745', '#ffc107', '#dc3545']
    
    # Create scatter plot
    for i in range(3):
        segment_data = df[df['segment'] == i]
        if len(segment_data) > 0:
            plt.scatter(segment_data['tenure'], segment_data['monthly_charges'], 
                       c=colors[i], label=f'Segment {i+1}', alpha=0.6, s=50, edgecolors='white', linewidth=1)
    
    plt.xlabel('Tenure (months)', fontsize=12)
    plt.ylabel('Monthly Charges ($)', fontsize=12)
    plt.title('Customer Segments Visualization', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add legend
    plt.legend()
    
    # Add risk level annotations
    plt.text(0.02, 0.98, '🟢 Low Risk', transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
    plt.text(0.02, 0.93, '🟡 Medium Risk', transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
    plt.text(0.02, 0.88, '🔴 High Risk', transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
    
    # Save plot to base64 string
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    
    return base64.b64encode(image_png).decode()

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Starting Churn Prediction Web App")
    print("="*60)
    print(f"\n📊 Models loaded:")
    print(f"   - Logistic Regression: {'✅' if models.get('logistic') else '❌'}")
    print(f"   - Decision Tree: {'✅' if models.get('tree') else '❌'}")
    print(f"   - Random Forest: {'✅' if models.get('forest') else '❌'}")
    print(f"\n📊 Feature count: {len(FEATURE_NAMES) if FEATURE_NAMES else 'Unknown'}")
    print(f"\n📜 History file: {HISTORY_FILE}")
    print(f"   Loaded {len(recent_predictions)} predictions from history")
    print("\n🌐 Open http://localhost:5000 in your browser")
    print("   📍 Landing Page: http://localhost:5000/")
    print("   🚀 App: http://localhost:5000/app")
    print("="*60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)