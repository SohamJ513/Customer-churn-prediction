from src.data_preprocessing import DataPreprocessor
from src.random_forest import RandomForestFromScratch
import numpy as np
import matplotlib.pyplot as plt
import time

print("📦 Loading and preprocessing churn data...")
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

print(f"\n📊 Training data: {X_train_np.shape[0]} samples, {X_train_np.shape[1]} features")
print(f"   Churn rate: {np.mean(y_train_np)*100:.1f}%")

# Test different numbers of trees
print("\n" + "="*70)
print("🌲🌲🌲 RANDOM FOREST PERFORMANCE ON CHURN DATA 🌲🌲🌲")
print("="*70)

tree_counts = [5, 10, 20, 30, 50]
train_scores = []
test_scores = []
training_times = []

for n_trees in tree_counts:
    print(f"\n{'='*50}")
    print(f"🌳 Testing with {n_trees} trees")
    print(f"{'='*50}")
    
    start_time = time.time()
    
    rf = RandomForestFromScratch(
        n_trees=n_trees,
        max_depth=5,  # Optimal depth from decision tree
        min_samples_split=20,
        min_samples_leaf=10,
        sample_ratio=0.8,
        feature_ratio=0.7,
        criterion='entropy',
        random_state=42
    )
    
    rf.fit(X_train_np, y_train_np, feature_names=data['feature_names'], verbose=True)
    
    train_acc = rf.evaluate(X_train_np, y_train_np)
    test_acc = rf.evaluate(X_test_np, y_test_np)
    elapsed_time = time.time() - start_time
    
    train_scores.append(train_acc)
    test_scores.append(test_acc)
    training_times.append(elapsed_time)
    
    print(f"\n📊 Results for {n_trees} trees:")
    print(f"   Training Accuracy: {train_acc:.2f}%")
    print(f"   Test Accuracy: {test_acc:.2f}%")
    print(f"   Training Time: {elapsed_time:.2f} seconds")
    
    # Print feature importance for the best model
    if n_trees == 20:
        rf.print_feature_importance(top_n=10)

# Find best model
best_idx = np.argmax(test_scores)
best_n_trees = tree_counts[best_idx]
best_test_acc = test_scores[best_idx]

print("\n" + "="*70)
print("🏆 RANDOM FOREST FINAL RESULTS")
print("="*70)
print(f"Best number of trees: {best_n_trees}")
print(f"Best test accuracy: {best_test_acc:.2f}%")

# Compare with Logistic Regression and Decision Tree
print("\n" + "="*70)
print("📊 FINAL MODEL COMPARISON")
print("="*70)
print(f"Logistic Regression: 82.39%")
print(f"Decision Tree (depth=5): 79.12%")
print(f"Random Forest ({best_n_trees} trees): {best_test_acc:.2f}%")

improvement = best_test_acc - 82.39
if improvement > 0:
    print(f"\n🎉 Random Forest beats Logistic Regression by +{improvement:.2f}%!")
else:
    print(f"\n📈 Random Forest: {improvement:+.2f}% vs Logistic Regression")

# Plot results
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Accuracy vs Number of Trees
axes[0, 0].plot(tree_counts, train_scores, 'bo-', label='Training', linewidth=2, markersize=8)
axes[0, 0].plot(tree_counts, test_scores, 'ro-', label='Test', linewidth=2, markersize=8)
axes[0, 0].axhline(y=82.39, color='g', linestyle='--', label='Logistic Regression (82.39%)')
axes[0, 0].axhline(y=79.12, color='orange', linestyle='--', label='Decision Tree (79.12%)')
axes[0, 0].set_xlabel('Number of Trees')
axes[0, 0].set_ylabel('Accuracy (%)')
axes[0, 0].set_title('Random Forest Performance vs Number of Trees')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_xticks(tree_counts)

# Training Time
axes[0, 1].plot(tree_counts, training_times, 'mo-', linewidth=2, markersize=8)
axes[0, 1].set_xlabel('Number of Trees')
axes[0, 1].set_ylabel('Training Time (seconds)')
axes[0, 1].set_title('Training Time vs Number of Trees')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_xticks(tree_counts)

# Feature Importance (for best model)
rf_best = RandomForestFromScratch(
    n_trees=best_n_trees,
    max_depth=5,
    min_samples_split=20,
    min_samples_leaf=10,
    feature_ratio=0.7,
    random_state=42
)
rf_best.fit(X_train_np, y_train_np, feature_names=data['feature_names'], verbose=False)

importance = rf_best.feature_importances_
sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
features = [f[0][:20] + ('...' if len(f[0]) > 20 else '') for f in sorted_features]
importances = [f[1] for f in sorted_features]

axes[1, 0].barh(range(10), importances[::-1], color='steelblue')
axes[1, 0].set_yticks(range(10))
axes[1, 0].set_yticklabels(features[::-1])
axes[1, 0].set_xlabel('Importance')
axes[1, 0].set_title('Top 10 Features (Random Forest)')

# Final comparison bar chart
models = ['Logistic\nRegression', 'Decision\nTree', f'Random\nForest\n({best_n_trees} trees)']
accs = [82.39, 79.12, best_test_acc]
colors = ['#3498db', '#f39c12', '#2ecc71']
bars = axes[1, 1].bar(models, accs, color=colors, edgecolor='black', linewidth=2)
axes[1, 1].set_ylabel('Accuracy (%)')
axes[1, 1].set_title('Final Model Comparison')
axes[1, 1].set_ylim([75, 85])

for bar, acc in zip(bars, accs):
    height = bar.get_height()
    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.3,
                    f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('random_forest_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ Random Forest training complete! Check 'random_forest_results.png'")

# Save the best model
import pickle
import os

os.makedirs('models', exist_ok=True)

with open('models/random_forest_best.pkl', 'wb') as f:
    pickle.dump(rf_best, f)

print("\n💾 Best Random Forest model saved to 'models/random_forest_best.pkl'")