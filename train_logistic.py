from src.data_preprocessing import DataPreprocessor
from src.logistic_regression import LogisticRegressionFromScratch
import matplotlib.pyplot as plt

# Load and preprocess data
print("📦 Loading and preprocessing data...")
preprocessor = DataPreprocessor('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
preprocessor.clean_data().encode_features().scale_features().prepare_train_test()

data = preprocessor.get_data()
X_train, X_test = data['X_train'], data['X_test']
y_train, y_test = data['y_train'], data['y_test']

# Train model
print("\n🤖 Training model...")
model = LogisticRegressionFromScratch(learning_rate=0.1, epochs=1000)
model.train(X_train, y_train, verbose=True)

# Evaluate on test set
print("\n📊 Evaluating on test set...")
test_accuracy = model.evaluate(X_test, y_test)
train_accuracy = model.evaluate(X_train, y_train)

print(f"\n🎯 Final Results:")
print(f"   Training Accuracy: {train_accuracy:.2f}%")
print(f"   Test Accuracy: {test_accuracy:.2f}%")

# Plot loss history
plt.figure(figsize=(10, 6))
plt.plot(model.loss_history)
plt.title('Training Loss Over Time')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('training_loss.png')
plt.show()

print("\n✅ Model training complete! Check 'training_loss.png' for the loss curve.")