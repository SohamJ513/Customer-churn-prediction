import pickle
import numpy as np
import os

print("="*60)
print("🔬 TESTING MODELS WITH 30 FEATURES")
print("="*60)

# Load the feature names
try:
    with open('models/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    print(f"✅ Loaded {len(feature_names)} feature names")
    print(f"   First 10: {feature_names[:10]}")
except Exception as e:
    print(f"❌ Error loading feature names: {e}")
    exit()

# Create a test sample with ALL 30 features (initialize with zeros)
test_features = np.zeros(len(feature_names))

# Create a HIGH RISK customer (tenure=2, electronic check, fiber optic, etc.)
high_risk_config = {
    'gender': 1,  # Male
    'tenure': 2,  # 2 months (HIGH RISK!)
    'PhoneService': 1,
    'InternetService_Fiber optic': 1,
    'PaperlessBilling': 1,
    'PaymentMethod_Electronic check': 1,
    # Month-to-month contract is default (no one-year or two-year checked)
}

# Set the values
for i, feature in enumerate(feature_names):
    if feature in high_risk_config:
        test_features[i] = high_risk_config[feature]
    elif feature in ['Contract_One year', 'Contract_Two year']:
        test_features[i] = 0  # Explicitly set to 0 (month-to-month)
    elif 'No' in feature or 'No internet' in feature:
        test_features[i] = 0  # Set all "No" options to 0

X_test = test_features.reshape(1, -1)

print(f"\n📊 Test Input: {len(test_features)} features")
print(f"   tenure = {test_features[feature_names.index('tenure')]} months")
print(f"   PaymentMethod_Electronic check = {test_features[feature_names.index('PaymentMethod_Electronic check')]}")
print(f"   InternetService_Fiber optic = {test_features[feature_names.index('InternetService_Fiber optic')]}")
print(f"   Feature vector shape: {X_test.shape}")

# Test each model
model_files = [
    ('optimized_model.pkl', 'Logistic Regression'),
    ('decision_tree_best.pkl', 'Decision Tree'),
    ('random_forest_best.pkl', 'Random Forest')
]

for model_file, model_name in model_files:
    print(f"\n{'-'*50}")
    print(f"📁 Testing {model_name} ({model_file})")
    print(f"{'-'*50}")
    
    file_path = os.path.join('models', model_file)
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        continue
    
    try:
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        print(f"✅ Model loaded successfully")
        print(f"   Model type: {type(model).__name__}")
        
        # Check if it's a dict (Logistic Regression saved incorrectly)
        if isinstance(model, dict):
            print(f"⚠️  Logistic Regression is saved as dict, not model object")
            print(f"   Attempting to reconstruct...")
            
            # Try to import your Logistic Regression class
            from src.logistic_regression import LogisticRegressionFromScratch
            
            # Create a new model and set its parameters
            new_model = LogisticRegressionFromScratch()
            if 'weights' in model:
                new_model.weights = model['weights']
                new_model.bias = model['bias']
                print(f"   ✅ Reconstructed Logistic Regression model")
                model = new_model
            else:
                print(f"   ❌ Cannot reconstruct - missing weights")
                continue
        
        # Try prediction
        if hasattr(model, 'predict'):
            pred = model.predict(X_test)[0]
            print(f"\n   🔮 Prediction: {'CHURN' if pred == 1 else 'STAY'}")
        
        # Try probability
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_test)[0]
            print(f"   📊 Class 0 (STAY): {proba[0]*100:.2f}%")
            print(f"   📊 Class 1 (CHURN): {proba[1]*100:.2f}%")
            print(f"   📊 Confidence: {max(proba)*100:.2f}%")
            
            # Check if it's using fallback
            if abs(proba[1]*100 - 65) < 0.1:
                print(f"   ⚠️  This looks like FALLBACK value")
            else:
                print(f"   ✅ This is REAL model prediction")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*60)
print("✅ Test complete!")