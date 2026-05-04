import pickle
import os
from src.decision_tree import DecisionTreeFromScratch
from src.data_preprocessing import DataPreprocessor
import numpy as np

print("📦 Loading and preprocessing data...")
preprocessor = DataPreprocessor('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
preprocessor.clean_data().encode_features().scale_features().prepare_train_test()

data = preprocessor.get_data()
X_train, X_test = data['X_train'], data['X_test']
y_train, y_test = data['y_train'], data['y_test']

# Convert to numpy
X_train_np = np.array(X_train)
X_test_np = np.array(X_test)
y_train_np = np.array(y_train)
y_test_np = np.array(y_test)

print("🌳 Testing different depths to find best model...")

best_accuracy = 0
best_model = None
best_depth = 0

# Test different depths (like you did before)
depths = [3, 5, 7, 10]

for depth in depths:
    print(f"\n📏 Testing depth = {depth}")
    
    dt = DecisionTreeFromScratch(
        max_depth=depth,
        min_samples_split=20,
        min_samples_leaf=10,
        criterion='entropy'
    )
    
    dt.fit(X_train_np, y_train_np, feature_names=data['feature_names'])
    
    train_acc = dt.evaluate(X_train_np, y_train_np)
    test_acc = dt.evaluate(X_test_np, y_test_np)
    
    print(f"   Train accuracy: {train_acc:.2f}%")
    print(f"   Test accuracy: {test_acc:.2f}%")
    
    if test_acc > best_accuracy:
        best_accuracy = test_acc
        best_model = dt
        best_depth = depth

print(f"\n🏆 Best model: depth={best_depth} with {best_accuracy:.2f}% test accuracy")

# Save the best model
model_path = 'models/decision_tree_best.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(best_model, f)

print(f"💾 Best model saved to {model_path}")

# Verify
if os.path.exists(model_path):
    print(f"✅ File created successfully!")
    
    # Test loading it back
    with open(model_path, 'rb') as f:
        loaded_model = pickle.load(f)
    print(f"✅ Model loaded successfully for verification")
    print(f"   Loaded model depth: {loaded_model.max_depth}")