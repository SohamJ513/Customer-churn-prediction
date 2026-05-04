import numpy as np
import pandas as pd
from src.decision_tree import DecisionTreeFromScratch
from collections import Counter

class RandomForestFromScratch:
    def __init__(self, n_trees=10, max_depth=5, min_samples_split=20, 
                 min_samples_leaf=10, sample_ratio=0.8, feature_ratio=0.7, 
                 criterion='entropy', random_state=42):
        """
        Random Forest Classifier built from scratch
        
        Args:
            n_trees: Number of decision trees in the forest
            max_depth: Maximum depth for each tree
            min_samples_split: Minimum samples to split a node
            min_samples_leaf: Minimum samples in a leaf node
            sample_ratio: Fraction of samples to use for bootstrap sampling
            feature_ratio: Fraction of features to consider at each split
            criterion: Split criterion ('entropy' or 'gini')
            random_state: Random seed for reproducibility
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.sample_ratio = sample_ratio
        self.feature_ratio = feature_ratio
        self.criterion = criterion
        self.random_state = random_state
        self.trees = []
        self.feature_names = None
        self.feature_importances_ = None
        
        # Set random seed
        np.random.seed(random_state)
        
    def _bootstrap_sample(self, X, y):
        """
        Create a bootstrap sample (sampling with replacement)
        """
        n_samples = X.shape[0]
        n_bootstrap = int(n_samples * self.sample_ratio)
        
        # Sample indices with replacement
        indices = np.random.choice(n_samples, n_bootstrap, replace=True)
        
        # Out-of-bag indices (samples not selected)
        oob_indices = list(set(range(n_samples)) - set(indices))
        
        return X[indices], y[indices], indices, oob_indices
    
    def _get_random_features(self, n_features):
        """
        Randomly select a subset of features for a tree
        """
        n_features_split = max(1, int(n_features * self.feature_ratio))
        feature_indices = np.random.choice(n_features, n_features_split, replace=False)
        return feature_indices
    
    def fit(self, X, y, feature_names=None, verbose=True):
        """
        Train the random forest
        
        Args:
            X: Training features
            y: Training labels
            feature_names: Names of features
            verbose: Print progress
        """
        print("\n" + "="*70)
        print("🌲🌲🌲 RANDOM FOREST FROM SCRATCH 🌲🌲🌲")
        print("="*70)
        
        # Convert to numpy
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist() if feature_names is None else feature_names
            X = X.values.astype(np.float64)
        else:
            X = np.array(X, dtype=np.float64)
            self.feature_names = feature_names if feature_names else [f"F{i}" for i in range(X.shape[1])]
        
        y = np.array(y)
        
        n_samples, n_features = X.shape
        
        print(f"📊 Training data: {n_samples} samples, {n_features} features")
        print(f"🌳 Number of trees: {self.n_trees}")
        print(f"📏 Max depth per tree: {self.max_depth}")
        print(f"📊 Bootstrap ratio: {self.sample_ratio}")
        print(f"🎯 Feature ratio: {self.feature_ratio}")
        print(f"📐 Split criterion: {self.criterion}")
        
        # Store out-of-bag predictions for each sample
        oob_predictions = {i: [] for i in range(n_samples)}
        
        # Train each tree
        for i in range(self.n_trees):
            if verbose:
                print(f"\n🌳 Training tree {i+1}/{self.n_trees}")
            
            # Bootstrap sample
            X_bootstrap, y_bootstrap, bootstrap_idx, oob_idx = self._bootstrap_sample(X, y)
            
            # Random feature selection for this tree
            feature_indices = self._get_random_features(n_features)
            
            if verbose and i == 0:
                print(f"   Using {len(feature_indices)} random features")
            
            # Create and train decision tree
            tree = DecisionTreeFromScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                criterion=self.criterion
            )
            
            # Train on bootstrap sample with selected features
            tree.fit(X_bootstrap[:, feature_indices], y_bootstrap, 
                    feature_names=[self.feature_names[j] for j in feature_indices],
                    debug=False)
            
            # Store tree and its feature indices
            self.trees.append({
                'tree': tree,
                'feature_indices': feature_indices
            })
            
            # Make OOB predictions
            if len(oob_idx) > 0:
                X_oob = X[oob_idx][:, feature_indices]
                y_oob_pred = tree.predict(X_oob)
                
                for j, idx in enumerate(oob_idx):
                    oob_predictions[idx].append(y_oob_pred[j])
            
            if verbose and (i+1) % 5 == 0:
                print(f"   Progress: {i+1}/{self.n_trees} trees trained")
        
        # Calculate OOB score
        oob_accuracy = self._calculate_oob_score(y, oob_predictions)
        print(f"\n📊 Out-of-Bag (OOB) Score: {oob_accuracy:.2f}%")
        
        # Calculate feature importance
        self._calculate_feature_importance()
        
        print("\n✅ Random Forest training complete!")
        
        return self
    
    def _calculate_oob_score(self, y, oob_predictions):
        """
        Calculate Out-of-Bag accuracy score
        """
        correct = 0
        total = 0
        
        for idx, preds in oob_predictions.items():
            if len(preds) > 0:
                # Majority vote
                unique, counts = np.unique(preds, return_counts=True)
                majority_pred = unique[np.argmax(counts)]
                
                if majority_pred == y[idx]:
                    correct += 1
                total += 1
        
        if total > 0:
            return (correct / total) * 100
        return 0
    
    def _calculate_feature_importance(self):
        """
        Calculate feature importance by averaging importance across all trees
        """
        importance_dict = {name: [] for name in self.feature_names}
        
        for tree_info in self.trees:
            tree = tree_info['tree']
            feature_indices = tree_info['feature_indices']
            
            # Get importance from tree
            tree_importance = tree.get_feature_importance()
            
            # Map back to original feature indices
            for local_idx, global_idx in enumerate(feature_indices):
                feature_name = self.feature_names[global_idx]
                local_feature_name = tree.feature_names[local_idx]
                if local_feature_name in tree_importance:
                    importance_dict[feature_name].append(tree_importance[local_feature_name])
        
        # Average importance across trees
        self.feature_importances_ = {}
        for feature, values in importance_dict.items():
            if values:
                self.feature_importances_[feature] = np.mean(values)
            else:
                self.feature_importances_[feature] = 0
        
        # Normalize
        total = sum(self.feature_importances_.values())
        if total > 0:
            for feature in self.feature_importances_:
                self.feature_importances_[feature] /= total
    
    def predict(self, X):
        """
        Predict labels using majority voting
        """
        if isinstance(X, pd.DataFrame):
            X = X.values.astype(np.float64)
        else:
            X = np.array(X, dtype=np.float64)
        
        # Get predictions from all trees
        tree_predictions = []
        
        for tree_info in self.trees:
            tree = tree_info['tree']
            feature_indices = tree_info['feature_indices']
            
            # Select features for this tree
            X_subset = X[:, feature_indices]
            
            # Get predictions
            preds = tree.predict(X_subset)
            tree_predictions.append(preds)
        
        # Convert to array
        tree_predictions = np.array(tree_predictions)
        
        # Majority voting for each sample
        final_predictions = []
        for i in range(X.shape[0]):
            votes = tree_predictions[:, i]
            # Count votes
            vote_counts = Counter(votes)
            # Get majority class
            majority_class = vote_counts.most_common(1)[0][0]
            final_predictions.append(majority_class)
        
        return np.array(final_predictions)
    
    def predict_proba(self, X):
        """
        Predict class probabilities (average of tree probabilities)
        Returns probabilities in the format [P(class=0), P(class=1)]
        """
        if isinstance(X, pd.DataFrame):
            X = X.values.astype(np.float64)
        else:
            X = np.array(X, dtype=np.float64)
        
        # Get probability predictions from all trees
        all_probas = []
        
        for tree_info in self.trees:
            tree = tree_info['tree']
            feature_indices = tree_info['feature_indices']
            
            X_subset = X[:, feature_indices]
            probas = tree.predict_proba(X_subset)
            
            # Ensure each tree's probabilities are in the right format [P(0), P(1)]
            if probas is None:
                # Skip if no probabilities
                continue
            
            # Check if probas is 1D (single sample) or 2D (multiple samples)
            if len(probas.shape) == 1:
                probas = probas.reshape(1, -1)
            
            # Ensure each probability row has 2 elements
            fixed_probas = []
            for p in probas:
                if isinstance(p, (int, float)):
                    # Single number - assume it's probability of class 1
                    fixed_probas.append([1 - p, p])
                elif len(p) == 1:
                    # Single value in array - assume it's probability of that class
                    full_proba = [0.0, 0.0]
                    full_proba[int(p[0])] = 1.0
                    fixed_probas.append(full_proba)
                elif len(p) == 2:
                    # Already correct format
                    fixed_probas.append([float(p[0]), float(p[1])])
                else:
                    # Unexpected format - use 50/50
                    print(f"⚠️ Unexpected probability format from tree: {p}")
                    fixed_probas.append([0.5, 0.5])
            
            all_probas.append(np.array(fixed_probas))
        
        if len(all_probas) == 0:
            # No valid probabilities from any tree
            print("⚠️ No valid probabilities from any tree, using fallback")
            return np.array([[0.5, 0.5]] * X.shape[0])
        
        # Average probabilities across all trees
        avg_probas = np.mean(all_probas, axis=0)
        
        # Ensure final shape is (n_samples, 2)
        if len(avg_probas.shape) == 1:
            avg_probas = avg_probas.reshape(1, -1)
        
        return avg_probas
    
    def evaluate(self, X, y):
        """
        Calculate accuracy
        """
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y) * 100
        return accuracy
    
    def print_feature_importance(self, top_n=10):
        """
        Print top N most important features
        """
        if self.feature_importances_ is None:
            print("Feature importance not calculated yet")
            return
        
        print(f"\n🔝 Top {top_n} Most Important Features:")
        sorted_features = sorted(self.feature_importances_.items(), 
                                 key=lambda x: x[1], reverse=True)
        
        for i, (feature, importance) in enumerate(sorted_features[:top_n]):
            print(f"   {i+1}. {feature}: {importance:.4f}")
    
    def get_params(self):
        """
        Get model parameters
        """
        return {
            'n_trees': self.n_trees,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'sample_ratio': self.sample_ratio,
            'feature_ratio': self.feature_ratio,
            'criterion': self.criterion,
            'oob_score': getattr(self, 'oob_score_', None)
        }


# Quick test
if __name__ == "__main__":
    # Test with simple data
    X_test = np.array([[0,0], [0,1], [1,0], [1,1]])
    y_test = np.array([0, 1, 1, 0])  # XOR problem
    
    rf = RandomForestFromScratch(n_trees=5, max_depth=3)
    rf.fit(X_test, y_test, feature_names=['F1', 'F2'], verbose=True)
    
    preds = rf.predict(X_test)
    accuracy = rf.evaluate(X_test, y_test)
    print(f"\n✅ Random Forest accuracy on XOR: {accuracy:.2f}%")
    
    # Test predict_proba
    probas = rf.predict_proba(X_test)
    print(f"\n📊 Probabilities shape: {probas.shape}")
    print(f"   Sample probabilities: {probas}")