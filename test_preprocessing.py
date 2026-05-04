from src.data_preprocessing import DataPreprocessor

# Initialize and run preprocessing
preprocessor = DataPreprocessor('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
preprocessor.clean_data().encode_features().scale_features().prepare_train_test()

# Get the data
data = preprocessor.get_data()

print("\n" + "="*60)
print("🎯 PREPROCESSING COMPLETE! READY FOR MODELING")
print("="*60)
print(f"\nTraining features shape: {data['X_train'].shape}")
print(f"Training labels shape: {data['y_train'].shape}")
print(f"\nFirst 5 training samples:")
print(data['X_train'].head())

# Verify scaling worked
print(f"\n📊 Feature ranges after scaling:")
for col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
    if col in data['X_train'].columns:
        print(f"   {col}: min={data['X_train'][col].min():.3f}, max={data['X_train'][col].max():.3f}")