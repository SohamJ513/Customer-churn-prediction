import pandas as pd
import numpy as np

class DataPreprocessor:
    def __init__(self, filepath):
        """Initialize with data file path"""
        self.df = pd.read_csv(filepath)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def clean_data(self):
        """Handle missing values and data type issues"""
        print("Starting data cleaning...")
        
        # Drop customerID (not a feature)
        self.df = self.df.drop('customerID', axis=1)
        
        # Fix TotalCharges - convert empty strings to NaN then to numeric
        self.df['TotalCharges'] = self.df['TotalCharges'].replace(' ', np.nan)
        self.df['TotalCharges'] = pd.to_numeric(self.df['TotalCharges'])
        
        # Fill NaN TotalCharges with 0 (for new customers)
        self.df['TotalCharges'] = self.df['TotalCharges'].fillna(0)
        
        print(f"✅ Data cleaned. Shape: {self.df.shape}")
        print(f"   TotalCharges - min: ${self.df['TotalCharges'].min():.2f}, max: ${self.df['TotalCharges'].max():.2f}")
        return self
    
    def encode_features(self):
        """Convert categorical variables to numerical"""
        print("\nStarting feature encoding...")
        df_encoded = self.df.copy()
        
        # Binary encoding for Yes/No columns
        binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
        for col in binary_cols:
            df_encoded[col] = df_encoded[col].map({'Yes': 1, 'No': 0})
            print(f"   Encoded {col}: Yes=1, No=0")
        
        # Encode gender
        df_encoded['gender'] = df_encoded['gender'].map({'Male': 1, 'Female': 0})
        print(f"   Encoded gender: Male=1, Female=0")
        
        # Encode Churn (target variable)
        df_encoded['Churn'] = df_encoded['Churn'].map({'Yes': 1, 'No': 0})
        print(f"   Encoded Churn: Yes=1, No=0")
        
        # One-hot encoding for multi-category columns
        multi_cat_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 
                          'OnlineBackup', 'DeviceProtection', 'TechSupport',
                          'StreamingTV', 'StreamingMovies', 'Contract', 
                          'PaymentMethod']
        
        print(f"\n   One-hot encoding: {multi_cat_cols}")
        df_encoded = pd.get_dummies(df_encoded, columns=multi_cat_cols, drop_first=True)
        
        self.df = df_encoded
        print(f"✅ Encoding complete. New shape: {self.df.shape}")
        print(f"   Features now: {self.df.columns.tolist()[:10]}... (showing first 10)")
        return self
    
    def scale_features(self):
        """Manual min-max scaling (from scratch)"""
        print("\nStarting feature scaling...")
        
        # Identify numerical columns to scale
        num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        
        scaling_params = {}
        for col in num_cols:
            min_val = self.df[col].min()
            max_val = self.df[col].max()
            
            # Store parameters for later use (e.g., on test data)
            scaling_params[col] = {'min': min_val, 'max': max_val}
            
            # Min-max scaling: (x - min) / (max - min)
            self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
            
            print(f"   Scaled {col}: min={min_val:.2f} → 0, max={max_val:.2f} → 1")
        
        print("✅ Feature scaling complete")
        return self
    
    def prepare_train_test(self, test_size=0.2, random_state=42):
        """Manual train-test split (without sklearn)"""
        print(f"\nSplitting data (test_size={test_size})...")
        
        # Separate features and target
        X = self.df.drop('Churn', axis=1)
        y = self.df['Churn']
        
        # Manual train-test split
        np.random.seed(random_state)
        indices = np.random.permutation(len(X))
        test_size = int(len(X) * test_size)
        test_indices = indices[:test_size]
        train_indices = indices[test_size:]
        
        self.X_train = X.iloc[train_indices].reset_index(drop=True)
        self.X_test = X.iloc[test_indices].reset_index(drop=True)
        self.y_train = y.iloc[train_indices].reset_index(drop=True)
        self.y_test = y.iloc[test_indices].reset_index(drop=True)
        
        print(f"✅ Split complete:")
        print(f"   Training set: {len(self.X_train)} samples ({len(self.X_train)/len(X)*100:.1f}%)")
        print(f"   Test set: {len(self.X_test)} samples ({len(self.X_test)/len(X)*100:.1f}%)")
        print(f"   Training churn rate: {self.y_train.mean()*100:.1f}%")
        print(f"   Test churn rate: {self.y_test.mean()*100:.1f}%")
        
        return self
    
    def get_feature_names(self):
        """Return list of feature names"""
        return self.X_train.columns.tolist() if self.X_train is not None else []
    
    def get_data(self):
        """Return all prepared data"""
        return {
            'X_train': self.X_train,
            'X_test': self.X_test,
            'y_train': self.y_train,
            'y_test': self.y_test,
            'feature_names': self.get_feature_names()
        }