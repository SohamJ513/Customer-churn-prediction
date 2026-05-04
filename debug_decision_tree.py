from src.decision_tree import DecisionTreeFromScratch
import numpy as np

print("🔍 DEBUGGING DECISION TREE")
print("="*60)

# Create an even simpler dataset - AND problem (easier than XOR)
X_and = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=np.float64)
y_and = np.array([0, 0, 0, 1])  # AND: only [1,1] is 1

print("\n📊 AND dataset:")
for i in range(len(X_and)):
    print(f"   {X_and[i]} -> {y_and[i]}")

# Train tree
dt = DecisionTreeFromScratch(max_depth=2, min_samples_split=2, min_samples_leaf=1)
dt.fit(X_and, y_and, feature_names=['Feature_1', 'Feature_2'])

print("\n🌳 Decision Tree Structure:")
dt.print_tree()

accuracy = dt.evaluate(X_and, y_and)
print(f"\n✅ Accuracy: {accuracy:.2f}%")

print("\n" + "="*60)
print("🔬 Testing XOR again with debug output")

# XOR dataset
X_xor = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=np.float64)
y_xor = np.array([0, 1, 1, 0])

print("\n📊 XOR dataset:")
for i in range(len(X_xor)):
    print(f"   {X_xor[i]} -> {y_xor[i]}")

# Train tree with debug output
dt_xor = DecisionTreeFromScratch(max_depth=3, min_samples_split=2, min_samples_leaf=1)
dt_xor.fit(X_xor, y_xor, feature_names=['Feature_1', 'Feature_2'])

print("\n🌳 Decision Tree Structure:")
dt_xor.print_tree()

accuracy = dt_xor.evaluate(X_xor, y_xor)
print(f"\n✅ Accuracy: {accuracy:.2f}%")