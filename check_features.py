import pickle
import numpy as np
import os

# Load your preprocessing module to see how features were created
from src.data_preprocessing import DataPreprocessor

print("📦 Loading preprocessing to see feature count...")
preprocessor = DataPreprocessor('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
preprocessor.clean_data().encode_features().scale_features().prepare_train_test()

data = preprocessor.get_data()
X_train = data['X_train']

print(f"\n✅ Training data has {X_train.shape[1]} features")
print(f"   Feature names: {list(X_train.columns[:10])}... (showing first 10)")

# Save the feature names for later use
feature_names = list(X_train.columns)
with open('models/feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)

print(f"\n💾 Saved {len(feature_names)} feature names to models/feature_names.pkl")