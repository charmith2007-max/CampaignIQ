
from model_loader import xgb_model, xgb_preprocessor
print("\n========== MODEL TEST ==========")
print("Model type:", type(xgb_model))
print("Preprocessor type:", type(xgb_preprocessor))
print("\nModel classes:", xgb_model.classes_)
print("\nModel and preprocessor loaded successfully!")