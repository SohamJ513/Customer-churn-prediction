from src.data_preprocessing import DataPreprocessor
from src.logistic_regression import LogisticRegressionFromScratch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# Load and preprocess data
print("📦 Loading and preprocessing data...")
preprocessor = DataPreprocessor('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
preprocessor.clean_data().encode_features().scale_features().prepare_train_test()

data = preprocessor.get_data()
X_train, X_test = data['X_train'], data['X_test']
y_train, y_test = data['y_train'], data['y_test']

# Train model with optimal parameters
print("\n🤖 Training final model...")
model = LogisticRegressionFromScratch(learning_rate=0.1, epochs=1000)
model.train(X_train, y_train, verbose=True)

# Get predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Calculate metrics
print("\n" + "="*60)
print("📊 DETAILED MODEL EVALUATION")
print("="*60)

# 1. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("\n🔷 Confusion Matrix:")
print(f"               Predicted")
print(f"              No    Yes")
print(f"Actual No    {tn:4d}   {fp:4d}")
print(f"       Yes   {fn:4d}   {tp:4d}")

# 2. Key Metrics
accuracy = (tp + tn) / (tp + tn + fp + fn) * 100
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n🎯 Performance Metrics:")
print(f"   Accuracy:  {accuracy:.2f}%")
print(f"   Precision: {precision:.3f} (of customers predicted to churn, {precision:.1%} actually did)")
print(f"   Recall:    {recall:.3f} (model caught {recall:.1%} of actual churners)")
print(f"   F1-Score:  {f1:.3f}")

# 3. Classification Report
print("\n📋 Detailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

# 4. Visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Loss curve
axes[0, 0].plot(model.loss_history)
axes[0, 0].set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Epochs')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].grid(True)

# Confusion Matrix Heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1],
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn'])
axes[0, 1].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Predicted')
axes[0, 1].set_ylabel('Actual')

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

axes[1, 0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
axes[1, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
axes[1, 0].set_xlim([0.0, 1.0])
axes[1, 0].set_ylim([0.0, 1.05])
axes[1, 0].set_xlabel('False Positive Rate')
axes[1, 0].set_ylabel('True Positive Rate')
axes[1, 0].set_title('ROC Curve', fontsize=14, fontweight='bold')
axes[1, 0].legend(loc="lower right")
axes[1, 0].grid(True)

# Feature Importance (weight magnitude)
weights = model.weights
feature_names = data['feature_names']
importance = np.abs(weights)
indices = np.argsort(importance)[-10:]  # Top 10 features

axes[1, 1].barh(range(len(indices)), importance[indices], color='steelblue')
axes[1, 1].set_yticks(range(len(indices)))
axes[1, 1].set_yticklabels([feature_names[i] for i in indices])
axes[1, 1].set_xlabel('|Weight| (Feature Importance)')
axes[1, 1].set_title('Top 10 Most Important Features', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ Evaluation complete! Check 'model_evaluation.png' for visualizations.")

# Save model parameters for later use
import pickle
model_params = {
    'weights': model.weights,
    'bias': model.bias,
    'accuracy': accuracy,
    'feature_names': feature_names
}

with open('models/trained_model.pkl', 'wb') as f:
    pickle.dump(model_params, f)
    
print("\n💾 Model saved to 'models/trained_model.pkl'")