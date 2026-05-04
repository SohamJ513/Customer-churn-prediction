import numpy as np
import pandas as pd

class LogisticRegressionFromScratch:
    def __init__(self, learning_rate=0.01, epochs=1000):
        """
        Custom Logistic Regression implementation
        
        Args:
            learning_rate: Step size for gradient descent
            epochs: Number of training iterations
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.loss_history = []
        
    def sigmoid(self, z):
        """
        Sigmoid activation function with improved numerical stability
        Formula: 1 / (1 + e^(-z))
        
        Args:
            z: Input array (can be numpy array, pandas series, or scalar)
        """
        # Convert to numpy array with float64 dtype
        z = np.array(z, dtype=np.float64)
        
        # Handle any NaN or inf values
        z = np.nan_to_num(z, nan=0.0, posinf=500, neginf=-500)
        
        # Clip to avoid overflow in exp
        z = np.clip(z, -500, 500)
        
        # Calculate sigmoid
        return 1 / (1 + np.exp(-z))
    
    def initialize_parameters(self, n_features):
        """
        Initialize weights and bias to zeros
        """
        self.weights = np.zeros(n_features)
        self.bias = 0
        print(f"   Initialized {n_features} weights and bias")
        
    def compute_cost(self, y_true, y_pred):
        """
        Binary Cross-Entropy Loss
        Formula: -1/m * sum(y * log(y_pred) + (1-y) * log(1-y_pred))
        
        Args:
            y_true: True labels
            y_pred: Predicted probabilities
        """
        m = len(y_true)
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Calculate cost
        cost = -(1/m) * np.sum(
            y_true * np.log(y_pred) + 
            (1 - y_true) * np.log(1 - y_pred)
        )
        
        # Check for NaN and return a large number if NaN
        if np.isnan(cost):
            return 1e10
        
        return cost
    
    def compute_gradients(self, X, y_true, y_pred):
        """
        Compute gradients for weights and bias
        
        Args:
            X: Feature matrix
            y_true: True labels
            y_pred: Predicted probabilities
        """
        m = X.shape[0]
        
        # Gradient for weights: (1/m) * X.T * (y_pred - y_true)
        dw = (1/m) * np.dot(X.T, (y_pred - y_true))
        
        # Gradient for bias: (1/m) * sum(y_pred - y_true)
        db = (1/m) * np.sum(y_pred - y_true)
        
        return dw, db
    
    def train(self, X, y, verbose=True):
        """
        Train the logistic regression model using gradient descent
        
        Args:
            X: Training features (n_samples, n_features) - can be pandas DataFrame
            y: Training labels (n_samples,) - can be pandas Series
            verbose: Print progress
        """
        print("\n" + "="*50)
        print("🚀 TRAINING LOGISTIC REGRESSION FROM SCRATCH")
        print("="*50)
        
        # Convert pandas objects to numpy arrays with proper dtype
        if isinstance(X, pd.DataFrame):
            X = X.values.astype(np.float64)
        else:
            X = np.array(X, dtype=np.float64)
            
        if isinstance(y, pd.Series):
            y = y.values.astype(np.float64)
        else:
            y = np.array(y, dtype=np.float64)
        
        n_samples, n_features = X.shape
        
        # Check for NaN or inf in input data
        if np.isnan(X).any():
            print("⚠️  Warning: NaN values found in X. Replacing with 0.")
            X = np.nan_to_num(X, nan=0.0)
        
        if np.isinf(X).any():
            print("⚠️  Warning: Inf values found in X. Replacing with large values.")
            X = np.nan_to_num(X, posinf=1e10, neginf=-1e10)
        
        # Initialize parameters
        self.initialize_parameters(n_features)
        
        # Gradient descent
        for epoch in range(self.epochs):
            # Forward pass: linear combination + sigmoid
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)
            
            # Compute cost
            cost = self.compute_cost(y, y_pred)
            self.loss_history.append(cost)
            
            # Compute gradients
            dw, db = self.compute_gradients(X, y, y_pred)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Print progress
            if verbose and epoch % 100 == 0:
                accuracy = self.evaluate(X, y)
                print(f"   Epoch {epoch:4d}, Loss: {cost:.4f}, Accuracy: {accuracy:.2f}%")
        
        print("\n✅ Training complete!")
        final_acc = self.evaluate(X, y)
        print(f"   Final training accuracy: {final_acc:.2f}%")
        
        return self
    
    def predict_proba(self, X):
        """
        Predict probability of class 1
        
        Args:
            X: Feature matrix
        """
        # Convert to numpy array
        if isinstance(X, pd.DataFrame):
            X = X.values.astype(np.float64)
        else:
            X = np.array(X, dtype=np.float64)
        
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)
    
    def predict(self, X, threshold=0.5):
        """
        Predict class labels (0 or 1)
        
        Args:
            X: Feature matrix
            threshold: Classification threshold (default 0.5)
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    
    def evaluate(self, X, y):
        """
        Calculate accuracy
        
        Args:
            X: Feature matrix
            y: True labels
        """
        # Convert y to numpy if needed
        if isinstance(y, pd.Series):
            y = y.values.astype(np.float64)
        else:
            y = np.array(y, dtype=np.float64)
        
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y) * 100
        return accuracy
    
    def get_params(self):
        """
        Return model parameters
        """
        return {
            'weights': self.weights,
            'bias': self.bias,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'loss_history': self.loss_history
        }
    
    def predict_with_confidence(self, X):
        """
        Predict both class and confidence score
        
        Args:
            X: Feature matrix
        """
        probabilities = self.predict_proba(X)
        predictions = (probabilities >= 0.5).astype(int)
        confidence = np.where(predictions == 1, probabilities, 1 - probabilities)
        return predictions, confidence