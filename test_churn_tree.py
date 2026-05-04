from src.data_preprocessing import DataPreprocessor
from src.decision_tree import DecisionTreeFromScratch
import numpy as np
import matplotlib.pyplot as plt

print("📦 Loading churn data...")
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

print(f"\n📊 Training on {len(X_train_np)} samples with {X_train_np.shape[1]} features")
print(f"   Churn rate: {np.mean(y_train_np)*100:.1f}%")

# Test different tree depths
print("\n" + "="*60)
print("🌳 TESTING DECISION TREE ON CHURN DATA")
print("="*60)

depths = [3, 5, 7, 10]
train_scores = []
test_scores = []

for depth in depths:
    print(f"\n📏 Testing max_depth = {depth}")
    
    dt = DecisionTreeFromScratch(
        max_depth=depth,
        min_samples_split=20,  # Prevent overfitting
        min_samples_leaf=10,    # Prevent overfitting
        criterion='entropy'
    )
    
    dt.fit(X_train_np, y_train_np, feature_names=data['feature_names'])
    
    train_acc = dt.evaluate(X_train_np, y_train_np)
    test_acc = dt.evaluate(X_test_np, y_test_np)
    
    train_scores.append(train_acc)
    test_scores.append(test_acc)
    
    print(f"   Training Accuracy: {train_acc:.2f}%")
    print(f"   Test Accuracy: {test_acc:.2f}%")
    
    # Print tree size
    print(f"   Tree depth: {dt._print_tree_summary()}")

# Find best depth
best_idx = np.argmax(test_scores)
best_depth = depths[best_idx]
print(f"\n🏆 Best depth: {best_depth} with {test_scores[best_idx]:.2f}% accuracy")

# Compare with Logistic Regression
print("\n" + "="*60)
print("📊 COMPARISON WITH LOGISTIC REGRESSION")
print("="*60)
print(f"Logistic Regression (from earlier): 82.39%")
print(f"Best Decision Tree: {test_scores[best_idx]:.2f}%")

# Plot comparison
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(depths, train_scores, 'bo-', label='Training', linewidth=2, markersize=8)
plt.plot(depths, test_scores, 'ro-', label='Test', linewidth=2, markersize=8)
plt.xlabel('Max Depth')
plt.ylabel('Accuracy (%)')
plt.title('Decision Tree Performance vs Depth')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
models = ['Logistic\nRegression', f'Decision Tree\n(depth={best_depth})']
accs = [82.39, test_scores[best_idx]]
colors = ['#3498db', '#2ecc71']
bars = plt.bar(models, accs, color=colors, edgecolor='black', linewidth=2)
plt.ylabel('Accuracy (%)')
plt.title('Model Comparison')
plt.ylim([75, 85])
for bar, acc in zip(bars, accs):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('churn_tree_comparison.png', dpi=300)
plt.show()

print("\n✅ Test complete! Check 'churn_tree_comparison.png'")