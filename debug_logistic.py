from src.data_preprocessing import DataPreprocessor
import numpy as np
import pandas as pd

# Load and preprocess data
print("📦 Loading and preprocessing data...")
preprocessor = DataPreprocessor('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
preprocessor.clean_data().encode_features().scale_features().prepare_train_test()

data = preprocessor.get_data()
X_train, X_test = data['X_train'], data['X_test']
y_train, y_test = data['y_train'], data['y_test']

# Debug: Check data types
print("\n🔍 DEBUGGING DATA:")
print(f"X_train type: {type(X_train)}")
print(f"X_train shape: {X_train.shape}")
print(f"X_train dtypes:")
print(X_train.dtypes.value_counts())
print(f"\nFirst row of X_train:")
print(X_train.iloc[0])
print(f"\nAny null values? {X_train.isnull().any().any()}")

# Try to convert to numpy and check
X_train_np = np.array(X_train, dtype=np.float64)
print(f"\nAfter conversion to numpy:")
print(f"Shape: {X_train_np.shape}")
print(f"dtype: {X_train_np.dtype}")
print(f"Any NaN? {np.isnan(X_train_np).any()}")
print(f"Min value: {X_train_np.min()}")
print(f"Max value: {X_train_np.max()}")

# Test sigmoid on a small sample
from src.logistic_regression import LogisticRegressionFromScratch
model = LogisticRegressionFromScratch()
test_sample = X_train_np[:5]
print(f"\n🧪 Testing sigmoid on sample:")
print(f"Sample shape: {test_sample.shape}")
result = model.sigmoid(test_sample @ np.zeros(X_train_np.shape[1]))
print(f"Sigmoid result: {result}")