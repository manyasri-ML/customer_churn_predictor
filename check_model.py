
import joblib

model_path = r"C:\Users\Manyasri rapelli\customer_churn_predictor\03_models\decision_tree_model.pkl"

model = joblib.load(model_path)

print("Model loaded successfully!")
print("Model type:", type(model))
print("Expected features:", getattr(model, "n_features_in_", "Not available"))
print("Feature names:", getattr(model, "feature_names_in_", "Not available"))

if hasattr(model, "steps"):
    print("Pipeline steps:")
    for name, step in model.steps:
        print(name, "->", type(step))
else:
    print("This is not a scikit-learn Pipeline.")
print("Feature names:", getattr(model, "feature_names_in_", None))
print("Classes:", model.classes_)