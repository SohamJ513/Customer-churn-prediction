import pickle
import numpy as np
import os

# Load the tree model
with open('models/decision_tree_best.pkl', 'rb') as f:
    tree = pickle.load(f)

print("="*60)
print("🔬 TESTING DECISION TREE PREDICT_PROBA DIRECTLY")
print("="*60)

# Create a test sample (low-risk customer)
test_features = np.zeros(30)  # 30 features

# Set values for low-risk customer
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

# Try to load feature names to map correctly
try:
    with open('models/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    print(f"✅ Loaded {len(feature_names)} feature names")
    
    for i, name in enumerate(feature_names):
        if name in feature_mapping:
            test_features[i] = feature_mapping[name]
        elif name == 'MonthlyCharges':
            test_features[i] = 140.0
        elif name == 'TotalCharges':
            test_features[i] = 140.0 * 72
        elif name == 'tenure':
            test_features[i] = feature_mapping['tenure']
    
    # Print some key feature values
    print("\n📊 Key feature values:")
    key_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract_Two year', 
                   'PaymentMethod_Credit card (automatic)', 'OnlineSecurity_Yes']
    for name in key_features:
        if name in feature_names:
            idx = feature_names.index(name)
            print(f"   {name:30} = {test_features[idx]}")
            
except Exception as e:
    print(f"⚠️ Could not load feature names: {e}")
    # Manual mapping if feature names not available
    test_features[4] = 72  # tenure
    test_features[16] = 1  # Contract_Two year
    test_features[19] = 1  # PaymentMethod_Credit card (automatic)

X_test = test_features.reshape(1, -1)
print(f"\n📊 Test input shape: {X_test.shape}")

# Test predict_proba
print("\n🔮 Testing tree.predict_proba()...")
proba = tree.predict_proba(X_test)
print(f"   Output: {proba}")
print(f"   Type: {type(proba)}")
print(f"   Shape: {proba.shape if hasattr(proba, 'shape') else 'N/A'}")
print(f"   Data type: {proba.dtype if hasattr(proba, 'dtype') else 'N/A'}")

if isinstance(proba, np.ndarray) and len(proba) > 0:
    print(f"   First sample: {proba[0]}")
    if len(proba[0]) >= 2:
        print(f"   Churn probability (class 1): {proba[0][1]*100:.2f}%")
        print(f"   Stay probability (class 0): {proba[0][0]*100:.2f}%")
        
        # Determine prediction from probabilities
        proba_pred = 1 if proba[0][1] > 0.5 else 0
        print(f"   Prediction from proba: {'CHURN' if proba_pred == 1 else 'STAY'}")

# Test predict
print("\n🔮 Testing tree.predict()...")
pred = tree.predict(X_test)
print(f"   Output: {pred}")
print(f"   Type: {type(pred)}")
print(f"   Shape: {pred.shape if hasattr(pred, 'shape') else 'N/A'}")
print(f"   Prediction: {'CHURN' if pred[0] == 1 else 'STAY'}")

# Compare the two
print("\n" + "="*60)
print("🔍 COMPARISON")
print("="*60)
print(f"predict() says:      {'CHURN' if pred[0] == 1 else 'STAY'}")
print(f"predict_proba() says: {'CHURN' if proba[0][1] > 0.5 else 'STAY'} (with {proba[0][1]*100:.2f}% churn probability)")

if (pred[0] == 1 and proba[0][1] > 0.5) or (pred[0] == 0 and proba[0][1] <= 0.5):
    print("✅ CONSISTENT: Both methods agree")
else:
    print("❌ INCONSISTENT: Methods disagree!")

# Examine tree structure
print("\n" + "="*60)
print("🔍 EXAMINING TREE STRUCTURE")
print("="*60)

def examine_tree(node, depth=0, path=""):
    indent = "  " * depth
    if node['type'] == 'leaf':
        probs = node.get('probabilities', 'N/A')
        print(f"{indent}🌿 Leaf at depth {depth}: class={node['class']}, probs={probs}, samples={node['samples']}")
        print(f"{indent}   Path: {path}")
    else:
        feature = node.get('feature_name', f'feature_{node["feature_idx"]}')
        print(f"{indent}🔀 Node at depth {depth}: {feature} <= {node['threshold']:.3f} (gain={node['gain']:.3f}, samples={node['samples']})")
        
        # Examine left child
        examine_tree(node['left'], depth + 1, path + f" → {feature} <= {node['threshold']:.3f}")
        
        # Examine right child
        examine_tree(node['right'], depth + 1, path + f" → {feature} > {node['threshold']:.3f}")

# Start examining from the root
examine_tree(tree.tree)

# Trace the path for our test sample
print("\n" + "="*60)
print("🔍 TRACING PATH FOR TEST SAMPLE")
print("="*60)

def trace_path(x, node, depth=0):
    if node['type'] == 'leaf':
        print(f"{'  ' * depth}🌿 REACHED LEAF: class={node['class']}, probs={node['probabilities']}")
        return node
    
    feature = node.get('feature_name', f'feature_{node["feature_idx"]}')
    feature_idx = node['feature_idx']
    threshold = node['threshold']
    value = x[feature_idx]
    
    if value <= threshold:
        print(f"{'  ' * depth}🔀 {feature} = {value:.3f} <= {threshold:.3f} → GO LEFT")
        return trace_path(x, node['left'], depth + 1)
    else:
        print(f"{'  ' * depth}🔀 {feature} = {value:.3f} > {threshold:.3f} → GO RIGHT")
        return trace_path(x, node['right'], depth + 1)

leaf_node = trace_path(test_features, tree.tree)
print(f"\n📌 Final leaf node probabilities: {leaf_node['probabilities']}")

print("\n" + "="*60)
print("✅ Test complete!")