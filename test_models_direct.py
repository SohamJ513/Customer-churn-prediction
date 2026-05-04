import pickle
import numpy as np
import os

print("="*60)
print("🔬 TESTING MODELS DIRECTLY")
print("="*60)

# Test case: High risk customer (tenure=2 months)
test_features = [1, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
X_test = np.array(test_features).reshape(1, -1)

print(f"\n📊 Test Input: tenure=2 months, electronic check=Yes")
print(f"   Features: {test_features[:10]}... (showing first 10)")

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
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        continue
    
    # Check file size
    file_size = os.path.getsize(file_path)
    print(f"📏 File size: {file_size} bytes")
    
    try:
        # Load model
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        print(f"✅ Model loaded successfully")
        print(f"   Model type: {type(model).__name__}")
        
        # Check what methods the model has
        print(f"   Methods available:")
        print(f"      - predict: {hasattr(model, 'predict')}")
        print(f"      - predict_proba: {hasattr(model, 'predict_proba')}")
        
        # Try prediction
        if hasattr(model, 'predict'):
            pred = model.predict(X_test)
            print(f"\n   🔮 Prediction result: {pred[0]}")
            print(f"   → {'CHURN' if pred[0] == 1 else 'STAY'}")
        else:
            print(f"   ❌ No predict method")
        
        # Try probability prediction
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_test)[0]
            print(f"\n   📊 Probability:")
            print(f"      - Class 0 (STAY): {proba[0]*100:.2f}%")
            print(f"      - Class 1 (CHURN): {proba[1]*100:.2f}%")
            print(f"      - Confidence: {max(proba)*100:.2f}%")
            
            # Check if it's using fallback (65%)
            if abs(proba[1]*100 - 65) < 0.1:
                print(f"   ⚠️  This looks like FALLBACK value (65%)")
            else:
                print(f"   ✅ This is REAL model prediction")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*60)
print("✅ Test complete!")