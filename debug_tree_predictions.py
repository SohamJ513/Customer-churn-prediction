import pickle
import numpy as np
import os

# Load feature names
with open('models/feature_names.pkl', 'rb') as f:
    FEATURE_NAMES = pickle.load(f)

# Load the tree model
with open('models/decision_tree_best.pkl', 'rb') as f:
    tree = pickle.load(f)

# Create a LOW RISK customer with ALL features
features = np.zeros(len(FEATURE_NAMES))

# Set the features for low-risk customer
feature_mapping = {
    'tenure': 72,
    'Contract_Two year': 1,
    'PaymentMethod_Credit card (automatic)': 1,
    'OnlineSecurity_Yes': 1,
    'OnlineBackup_Yes': 1,
    'DeviceProtection_Yes': 1,
    'TechSupport_Yes': 1,
    'StreamingTV_Yes': 1,
    'StreamingMovies_Yes': 1,
    'PhoneService': 1,
    'MultipleLines_Yes': 1,
    'gender': 1,
    'Partner': 1,
    'Dependents': 1,
    'PaperlessBilling': 1
}

# Calculate MonthlyCharges (using YOUR app's formula)
monthly = 20.0  # base
monthly += 20.0  # phone
monthly += 10.0  # multiple lines
monthly += 50.0  # DSL internet
monthly += 8.0 * 3  # online security, backup, protection
monthly += 15.0  # tech support
monthly += 12.0 * 2  # streaming TV & movies
monthly = min(monthly, 140.0)

print(f"💰 Calculated Monthly Charges: ${monthly:.2f}")

# Set values
for i, feature in enumerate(FEATURE_NAMES):
    if feature in feature_mapping:
        features[i] = feature_mapping[feature]
    elif feature == 'MonthlyCharges':
        features[i] = monthly
    elif feature == 'TotalCharges':
        features[i] = monthly * 72

# Try DIFFERENT scaling approaches
print("\n🔬 TESTING DIFFERENT SCALING APPROACHES:")
print("="*60)

# Approach 1: No scaling
X1 = features.copy().reshape(1, -1)
pred1 = tree.predict(X1)[0]
proba1 = tree.predict_proba(X1)[0]
print(f"\n1️⃣ NO SCALING:")
print(f"   Prediction: {'CHURN' if pred1 == 1 else 'STAY'}")
print(f"   Churn Prob: {proba1[1]*100:.2f}%")

# Approach 2: Scale tenure only
X2 = features.copy()
tenure_idx = FEATURE_NAMES.index('tenure')
X2[tenure_idx] = X2[tenure_idx] / 72.0
X2 = X2.reshape(1, -1)
pred2 = tree.predict(X2)[0]
proba2 = tree.predict_proba(X2)[0]
print(f"\n2️⃣ SCALE TENURE ONLY:")
print(f"   Prediction: {'CHURN' if pred2 == 1 else 'STAY'}")
print(f"   Churn Prob: {proba2[1]*100:.2f}%")

# Approach 3: Scale all numerical (as in your app)
X3 = features.copy()
X3[FEATURE_NAMES.index('tenure')] = X3[FEATURE_NAMES.index('tenure')] / 72.0
X3[FEATURE_NAMES.index('MonthlyCharges')] = X3[FEATURE_NAMES.index('MonthlyCharges')] / 140.0
X3[FEATURE_NAMES.index('TotalCharges')] = X3[FEATURE_NAMES.index('TotalCharges')] / 10000.0
X3 = X3.reshape(1, -1)
pred3 = tree.predict(X3)[0]
proba3 = tree.predict_proba(X3)[0]
print(f"\n3️⃣ FULL SCALING (your current approach):")
print(f"   Prediction: {'CHURN' if pred3 == 1 else 'STAY'}")
print(f"   Churn Prob: {proba3[1]*100:.2f}%")