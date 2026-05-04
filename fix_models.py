import pickle
import os

print("📁 Checking models directory...")
print(f"Files found: {os.listdir('models')}")

if 'trained_model.pkl' in os.listdir('models'):
    print("📦 Loading trained_model.pkl...")
    with open('models/trained_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    print("💾 Saving as optimized_model.pkl...")
    with open('models/optimized_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("✅ Created optimized_model.pkl from trained_model.pkl")
    
    # Also check if it worked
    if 'optimized_model.pkl' in os.listdir('models'):
        print("✅ Verification: optimized_model.pkl now exists!")
else:
    print("❌ trained_model.pkl not found")
    print("\nAvailable model files:")
    for file in os.listdir('models'):
        if file.endswith('.pkl'):
            print(f"   - {file}")