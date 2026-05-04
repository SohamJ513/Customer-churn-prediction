import numpy as np
import pandas as pd

class DecisionTreeFromScratch:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, criterion='entropy'):
        """
        Decision Tree Classifier built from scratch
        
        Args:
            max_depth: Maximum depth of the tree (None for unlimited)
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf: Minimum samples required in a leaf node
            criterion: Split criterion ('entropy' or 'gini')
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.tree = None
        self.n_features = None
        self.feature_names = None
        self.debug = False  # Set to True to see debug output
        
    def _entropy(self, y):
        """Calculate entropy of a label set"""
        if len(y) == 0:
            return 0
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))
    
    def _gini(self, y):
        """Calculate Gini impurity of a label set"""
        if len(y) == 0:
            return 0
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)
    
    def _impurity(self, y):
        """Calculate impurity based on chosen criterion"""
        if self.criterion == 'entropy':
            return self._entropy(y)
        else:
            return self._gini(y)
    
    def _information_gain(self, y, y_left, y_right):
        """Calculate information gain from a split"""
        if len(y_left) == 0 or len(y_right) == 0:
            return 0
        
        parent_impurity = self._impurity(y)
        n = len(y)
        n_left = len(y_left)
        n_right = len(y_right)
        
        child_impurity = (n_left/n) * self._impurity(y_left) + (n_right/n) * self._impurity(y_right)
        return parent_impurity - child_impurity
    
    def _find_best_split(self, X, y):
        """
        Find the best feature and threshold to split on (DEBUG VERSION)
        """
        best_gain = -1
        best_feature_idx = None
        best_threshold = None
        
        n_samples, n_features = X.shape
        
        if self.debug:
            print(f"\n🔍 Finding best split for {n_samples} samples")
            print(f"   Classes: {np.unique(y, return_counts=True)}")
        
        # If node is pure or too small, don't split
        if len(np.unique(y)) == 1 or n_samples < self.min_samples_split:
            if self.debug:
                print(f"   ⚠️ Skipping: pure node or too small")
            return best_feature_idx, best_threshold, best_gain
        
        # Try each feature
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            
            if self.debug:
                print(f"\n   📊 Feature {feature_idx}: {feature_values}")
            
            # Get unique values
            unique_values = np.unique(feature_values)
            
            if self.debug:
                print(f"      Unique values: {unique_values}")
            
            # Try every possible split point
            for i in range(len(unique_values)):
                threshold = unique_values[i]
                
                if self.debug:
                    print(f"      Testing threshold: {threshold}")
                
                # Split data
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                left_count = np.sum(left_mask)
                right_count = np.sum(right_mask)
                
                if self.debug:
                    print(f"         Left: {left_count} samples, Right: {right_count} samples")
                
                # Skip if split is invalid
                if left_count == 0 or right_count == 0:
                    if self.debug:
                        print(f"         ⚠️ Invalid split - one side empty")
                    continue
                
                y_left = y[left_mask]
                y_right = y[right_mask]
                
                if self.debug:
                    print(f"         Left classes: {np.unique(y_left, return_counts=True)}")
                    print(f"         Right classes: {np.unique(y_right, return_counts=True)}")
                
                # Skip if split creates too small leaves
                if left_count < self.min_samples_leaf or right_count < self.min_samples_leaf:
                    if self.debug:
                        print(f"         ⚠️ Leaf too small (min_samples_leaf={self.min_samples_leaf})")
                    continue
                
                # Calculate information gain
                gain = self._information_gain(y, y_left, y_right)
                
                if self.debug:
                    print(f"         Gain: {gain:.6f}")
                
                # Update best split if this is better
                if gain > best_gain:
                    best_gain = gain
                    best_feature_idx = feature_idx
                    best_threshold = threshold
                    if self.debug:
                        print(f"         ✅ New best! Gain: {gain:.6f}")
        
        if self.debug and best_feature_idx is not None:
            print(f"\n✅ Best split: Feature {best_feature_idx} at {best_threshold} with gain {best_gain:.6f}")
        elif self.debug:
            print(f"\n❌ No good split found")
        
        return best_feature_idx, best_threshold, best_gain
    
    def _build_tree(self, X, y, depth=0):
        """Recursively build the decision tree"""
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Check stopping criteria
        stop_criteria = False
        stop_reason = ""
        
        if self.max_depth is not None and depth >= self.max_depth:
            stop_criteria = True
            stop_reason = f"max_depth reached ({depth})"
        elif n_classes == 1:
            stop_criteria = True
            stop_reason = "pure node"
        elif n_samples < self.min_samples_split:
            stop_criteria = True
            stop_reason = f"insufficient samples ({n_samples} < {self.min_samples_split})"
        
        if stop_criteria:
            # Create leaf node with majority class and class probabilities
            unique, counts = np.unique(y, return_counts=True)
            majority_class = unique[np.argmax(counts)]
            probabilities = counts / n_samples
            
            # Ensure probabilities are stored as a list/array with 2 values for binary classification
            if len(probabilities) == 1:
                # If only one class, create 2-element array with probability 1 for that class
                full_probs = np.zeros(2)
                full_probs[int(majority_class)] = 1.0
                probabilities = full_probs
            
            if self.debug:
                print(f"{'  ' * depth}📌 Creating leaf at depth {depth}: class {majority_class} ({stop_reason})")
                print(f"{'  ' * depth}   Probabilities: {probabilities}")
            
            return {
                'type': 'leaf',
                'class': majority_class,
                'probabilities': probabilities,
                'samples': n_samples,
                'depth': depth,
                'stop_reason': stop_reason
            }
        
        # Find best split
        if self.debug:
            print(f"{'  ' * depth}🔍 Finding split at depth {depth}")
        
        best_feature_idx, best_threshold, best_gain = self._find_best_split(X, y)
        
        # If no good split found, create leaf
        if best_feature_idx is None or best_gain <= 1e-10:
            unique, counts = np.unique(y, return_counts=True)
            majority_class = unique[np.argmax(counts)]
            probabilities = counts / n_samples
            
            # Ensure probabilities are stored as a list/array with 2 values for binary classification
            if len(probabilities) == 1:
                full_probs = np.zeros(2)
                full_probs[int(majority_class)] = 1.0
                probabilities = full_probs
            
            if self.debug:
                print(f"{'  ' * depth}📌 Creating leaf at depth {depth}: no good split (gain={best_gain:.6f})")
                print(f"{'  ' * depth}   Probabilities: {probabilities}")
            
            return {
                'type': 'leaf',
                'class': majority_class,
                'probabilities': probabilities,
                'samples': n_samples,
                'depth': depth,
                'stop_reason': "no good split"
            }
        
        # Split the data
        feature_values = X[:, best_feature_idx]
        left_mask = feature_values <= best_threshold
        right_mask = ~left_mask
        
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]
        
        if self.debug:
            print(f"{'  ' * depth}🔀 Splitting on feature {best_feature_idx} at {best_threshold:.3f} (gain={best_gain:.6f})")
            print(f"{'  ' * depth}   Left: {len(y_left)} samples, Right: {len(y_right)} samples")
        
        # Recursively build children
        left_subtree = self._build_tree(X_left, y_left, depth + 1)
        right_subtree = self._build_tree(X_right, y_right, depth + 1)
        
        # Create decision node
        feature_name = self.feature_names[best_feature_idx] if self.feature_names else f"Feature_{best_feature_idx}"
        
        return {
            'type': 'node',
            'feature_idx': best_feature_idx,
            'feature_name': feature_name,
            'threshold': best_threshold,
            'gain': best_gain,
            'left': left_subtree,
            'right': right_subtree,
            'samples': n_samples,
            'depth': depth
        }
    
    def fit(self, X, y, feature_names=None, debug=False):
        """Train the decision tree"""
        self.debug = debug
        
        print("\n" + "="*60)
        print("🌳 TRAINING DECISION TREE FROM SCRATCH")
        print("="*60)
        
        # Convert to numpy
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist() if feature_names is None else feature_names
            X = X.values.astype(np.float64)
        else:
            X = np.array(X, dtype=np.float64)
            self.feature_names = feature_names if feature_names else [f"Feature_{i}" for i in range(X.shape[1])]
        
        y = np.array(y)
        
        self.n_features = X.shape[1]
        
        print(f"📊 Training data: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"📏 Max depth: {self.max_depth}")
        print(f"🔢 Min samples split: {self.min_samples_split}")
        print(f"🔢 Min samples leaf: {self.min_samples_leaf}")
        print(f"📐 Split criterion: {self.criterion}")
        print(f"🔍 Debug mode: {self.debug}")
        
        # Build the tree
        self.tree = self._build_tree(X, y)
        
        print("✅ Training complete!")
        self._print_tree_summary()
        
        return self
    
    def _predict_sample(self, x, node):
        """Predict a single sample by traversing the tree"""
        if node['type'] == 'leaf':
            return node['class']
        
        if x[node['feature_idx']] <= node['threshold']:
            return self._predict_sample(x, node['left'])
        else:
            return self._predict_sample(x, node['right'])
    
    def predict(self, X):
        """Predict labels for samples"""
        if isinstance(X, pd.DataFrame):
            X = X.values.astype(np.float64)
        else:
            X = np.array(X, dtype=np.float64)
        
        # Use predict_proba for consistency
        probas = self.predict_proba(X)
        predictions = [1 if p[1] > 0.5 else 0 for p in probas]
        return np.array(predictions)
    
    def predict_proba(self, X):
        """
        Predict class probabilities by traversing the tree
        Returns probabilities in the format [P(class=0), P(class=1)]
        """
        if isinstance(X, pd.DataFrame):
            X = X.values.astype(np.float64)
        else:
            X = np.array(X, dtype=np.float64)
        
        def _get_probabilities(x, node):
            if node['type'] == 'leaf':
                probs = node['probabilities']
                leaf_class = node['class']
                
                # Handle different probability formats
                if probs is None:
                    # Fallback if no probability
                    return np.array([0.5, 0.5])
                elif isinstance(probs, (int, float)):
                    # Single number - it's the probability for the leaf's class
                    full_probs = np.zeros(2)
                    full_probs[leaf_class] = probs
                    full_probs[1 - leaf_class] = 1 - probs
                    return full_probs
                elif len(probs) == 1:
                    # Single value in array - it's the probability for the leaf's class
                    full_probs = np.zeros(2)
                    full_probs[leaf_class] = probs[0]
                    full_probs[1 - leaf_class] = 1 - probs[0]
                    return full_probs
                elif len(probs) == 2:
                    # Already correct format - ensure it's a numpy array
                    return np.array([probs[0], probs[1]])
                else:
                    # Unexpected format - return probability based on leaf class
                    full_probs = np.zeros(2)
                    full_probs[leaf_class] = 1.0
                    return full_probs
            
            # Traverse the tree
            if x[node['feature_idx']] <= node['threshold']:
                return _get_probabilities(x, node['left'])
            else:
                return _get_probabilities(x, node['right'])
        
        probas = []
        for x in X:
            proba = _get_probabilities(x, self.tree)
            probas.append(proba)
        
        return np.array(probas)
    
    def evaluate(self, X, y):
        """Calculate accuracy"""
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y) * 100
        return accuracy
    
    def _print_tree_summary(self):
        """Print summary of the tree structure"""
        def count_nodes(node):
            if node['type'] == 'leaf':
                return 1
            return 1 + count_nodes(node['left']) + count_nodes(node['right'])
        
        def count_leaves(node):
            if node['type'] == 'leaf':
                return 1
            return count_leaves(node['left']) + count_leaves(node['right'])
        
        def get_max_depth(node):
            if node['type'] == 'leaf':
                return node['depth']
            return max(get_max_depth(node['left']), get_max_depth(node['right']))
        
        n_nodes = count_nodes(self.tree)
        n_leaves = count_leaves(self.tree)
        depth = get_max_depth(self.tree)
        
        print(f"\n📐 Tree Structure:")
        print(f"   Total nodes: {n_nodes}")
        print(f"   Leaf nodes: {n_leaves}")
        print(f"   Tree depth: {depth}")
    
    def print_tree(self, node=None, indent=""):
        """Pretty print the decision tree"""
        if node is None:
            node = self.tree
        
        if node['type'] == 'leaf':
            probs = node.get('probabilities', [1.0])
            prob_str = f" (probs: {np.round(probs, 3)})" if len(probs) > 1 else ""
            stop_reason = node.get('stop_reason', '')
            stop_str = f" [{stop_reason}]" if stop_reason else ""
            print(f"{indent}🌿 Class: {node['class']} [samples: {node['samples']}]{prob_str}{stop_str}")
        else:
            gain_str = f", gain: {node['gain']:.3f}"
            print(f"{indent}🔀 {node['feature_name']} <= {node['threshold']:.3f}{gain_str} [samples: {node['samples']}]")
            print(f"{indent}  ├─ Left:")
            self.print_tree(node['left'], indent + "  │ ")
            print(f"{indent}  └─ Right:")
            self.print_tree(node['right'], indent + "    ")
    
    def get_feature_importance(self):
        """Calculate feature importance based on gain and samples"""
        importance = {name: 0 for name in self.feature_names}
        
        def traverse(node):
            if node['type'] == 'node':
                importance[node['feature_name']] += node['gain'] * node['samples']
                traverse(node['left'])
                traverse(node['right'])
        
        traverse(self.tree)
        
        # Normalize
        total = sum(importance.values())
        if total > 0:
            for key in importance:
                importance[key] /= total
        
        return importance
    
    def test_with_simple_data(self, debug=False):
        """Test the decision tree with simple data"""
        print("\n" + "="*60)
        print("🧪 TESTING DECISION TREE WITH SIMPLE DATA")
        print("="*60)
        
        # Create simple dataset (XOR-like problem)
        X_simple = np.array([
            [1, 1],
            [1, 0],
            [0, 1],
            [0, 0]
        ], dtype=np.float64)
        y_simple = np.array([1, 0, 0, 1])
        
        print(f"\n📊 Simple dataset (XOR problem):")
        for i in range(len(X_simple)):
            arrow = "➡️" if X_simple[i, 0] == X_simple[i, 1] else "↗️"
            print(f"   {X_simple[i]} {arrow} {y_simple[i]}")
        
        # Train tree with different criteria
        for criterion in ['entropy', 'gini']:
            print(f"\n📐 Using {criterion} criterion:")
            dt = DecisionTreeFromScratch(
                max_depth=3, 
                min_samples_split=2, 
                min_samples_leaf=1, 
                criterion=criterion
            )
            dt.fit(X_simple, y_simple, feature_names=['Feature_1', 'Feature_2'], debug=debug)
            
            # Print tree
            print("\n🌳 Decision Tree Structure:")
            dt.print_tree()
            
            # Evaluate
            accuracy = dt.evaluate(X_simple, y_simple)
            print(f"\n✅ Accuracy on training data: {accuracy:.2f}%")
        
        return self
    
    def test_split_utilities(self):
        """Test the utility functions with simple examples"""
        print("\n" + "="*60)
        print("🧪 TESTING UTILITY FUNCTIONS")
        print("="*60)
        
        # Test entropy
        y_test = np.array([0, 0, 1, 1, 1, 1])
        print(f"\nTest labels: {y_test}")
        print(f"Entropy: {self._entropy(y_test):.4f} (should be ~0.9183)")
        print(f"Gini Impurity: {self._gini(y_test):.4f} (should be ~0.4444)")
        
        # Test information gain
        y_parent = np.array([0, 0, 1, 1, 1, 1, 0, 1])
        y_left = np.array([0, 0, 0])  # All 0's
        y_right = np.array([1, 1, 1, 1, 1])  # All 1's
        
        ig = self._information_gain(y_parent, y_left, y_right)
        print(f"\nPerfect split information gain: {ig:.4f}")
        
        # Test with impure split
        y_left_impure = np.array([0, 0, 1])
        y_right_impure = np.array([1, 1, 0, 0, 1])
        ig_impure = self._information_gain(y_parent, y_left_impure, y_right_impure)
        print(f"Impure split information gain: {ig_impure:.4f}")
        
        return self


# Quick test
if __name__ == "__main__":
    dt = DecisionTreeFromScratch(max_depth=3)
    dt.test_split_utilities()
    
    # Run with debug=False first, then change to True to see details
    dt.test_with_simple_data(debug=True)