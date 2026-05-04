from src.decision_tree import DecisionTreeFromScratch
import numpy as np

print("🌳 Testing Decision Tree Implementation")
print("="*50)

# Create instance
dt = DecisionTreeFromScratch(max_depth=3)

# Test utilities first
dt.test_split_utilities()

# Test with simple data
dt.test_with_simple_data()

print("\n✅ All tests complete!")