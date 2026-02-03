import joblib
import os

MODEL_PATH = os.path.join("results_article_style", "models", "stage3_best_features", "SVM.joblib")

m = joblib.load(MODEL_PATH)
print("Model type:", type(m))
print("Has attribute feature_names_in_:", hasattr(m, "feature_names_in_"))
if hasattr(m, "feature_names_in_"):
    print("feature_names_in_ (len):", len(getattr(m, "feature_names_in_")))
    print(getattr(m, "feature_names_in_"))

# If pipeline, inspect steps
if hasattr(m, "named_steps"):
    print("Pipeline steps:", list(m.named_steps.keys()))
    last = list(m.named_steps.values())[-1]
    print("Last estimator type:", type(last))
    print("Last has feature_names_in_:", hasattr(last, "feature_names_in_"))
    if hasattr(last, "feature_names_in_"):
        print("Last.feature_names_in_:", getattr(last, "feature_names_in_"))

# If estimator has coef_ or support_, print shapes to infer
if hasattr(m, "coef_"):
    try:
        print("coef_ shape:", getattr(m, "coef_").shape)
    except Exception:
        pass
if hasattr(m, "support_"):
    try:
        print("support_ len:", len(getattr(m, "support_")))
    except Exception:
        pass
