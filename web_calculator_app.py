from __future__ import annotations

import os
import json
from typing import Dict, Any, Tuple

import numpy as np
import joblib

# Avoid GUI backend issues
import matplotlib
matplotlib.use("Agg")  # must be before pyplot
import shap

from flask import Flask, request, jsonify, render_template_string

from common_config import MOD_DIR, log, shap_force_waterfall_svg

APP_TITLE = "AA 3-month Response Calculator (Binary-core STACK + SVM SHAP)"

MODEL_DIR = os.path.join(MOD_DIR, "binary_core_models")
CFG_PATH = os.path.join(MODEL_DIR, "binary_core_config.json")
STACK_PATH = os.path.join(MODEL_DIR, "stacking.joblib")
SVM_PATH = os.path.join(MODEL_DIR, "SVM.joblib")

if not (os.path.exists(CFG_PATH) and os.path.exists(STACK_PATH) and os.path.exists(SVM_PATH)):
    raise FileNotFoundError(
        "Missing model artifacts. Please run retrain_binary_core_models.py first."
    )

with open(CFG_PATH, "r", encoding="utf-8") as f:
    CFG = json.load(f)

FEATURE_ORDER = CFG["feature_order"]
THRESHOLDS = CFG.get("thresholds", {})
FIXED = CFG.get("fixed_binary_features", [])

# 显示名称映射（将内部特征名映射为用户友好的显示名）
DISPLAY_NAMES = {
    "Site_4.0": "Occipital",
}
def get_display_name(feature: str) -> str:
    return DISPLAY_NAMES.get(feature, feature)

STACK_MODEL = joblib.load(STACK_PATH)
SVM_MODEL = joblib.load(SVM_PATH)

# ---- Prepare Kernel SHAP background once (from saved binary train matrix)
TRAIN_BIN_CSV = CFG.get("train_binary_core_csv", "")
if not TRAIN_BIN_CSV or not os.path.exists(TRAIN_BIN_CSV):
    # fallback default location
    from common_config import DAT_DIR
    TRAIN_BIN_CSV = os.path.join(DAT_DIR, "binary_core", "X_train_binary_core.csv")

X_BG_ALL = np.loadtxt(TRAIN_BIN_CSV, delimiter=",", skiprows=1)
if X_BG_ALL.ndim == 1:
    X_BG_ALL = X_BG_ALL.reshape(1, -1)

rng = np.random.default_rng(19880216)
bg_n = min(100, X_BG_ALL.shape[0])
bg_idx = rng.choice(X_BG_ALL.shape[0], size=bg_n, replace=False) if X_BG_ALL.shape[0] > bg_n else np.arange(X_BG_ALL.shape[0])
X_BG = X_BG_ALL[bg_idx].astype(float)

# Build explainer once
_f = lambda z: SVM_MODEL.predict_proba(z)[:, 1]
SVM_EXPLAINER = shap.KernelExplainer(_f, X_BG)

# Where to write temporary SVGs (then embed the SVG text in HTML)
WEB_OUT_DIR = os.path.join(MODEL_DIR, "web_outputs")
os.makedirs(WEB_OUT_DIR, exist_ok=True)

app = Flask(__name__)

HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{{ title }}</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 28px; }
    .card { max-width: 980px; padding: 18px; border: 1px solid #ddd; border-radius: 10px; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px 18px; }
    label { font-weight: 600; }
    select { width: 100%; padding: 8px; }
    .btn { margin-top: 14px; padding: 10px 14px; border: 0; border-radius: 8px; cursor: pointer; }
    .btn { background: #1f4e79; color: white; }
    .note { color: #555; font-size: 13px; margin-top: 12px; line-height: 1.4; }
    .result { margin-top: 16px; padding: 12px; background: #f7fbff; border-left: 4px solid #1f4e79; }
    .small { font-size: 12px; color: #666; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; }
    .plots { margin-top: 16px; display: grid; grid-template-columns: 1fr; gap: 14px; }
    .plotbox { border: 1px solid #e5e5e5; border-radius: 10px; padding: 10px; background: white; overflow: auto; }
    .plotbox h4 { margin: 0 0 8px 0; }
  </style>
</head>
<body>
  <h2>{{ title }}</h2>

  <div class="card">
    <form method="post">
      <div class="grid">
        {% for f in features %}
        <div>
          <label for="{{ f }}">{{ display_names.get(f, f) }}</label><br/>
          <select id="{{ f }}" name="{{ f }}">
            <option value="0" {% if form.get(f, '0') == '0' %}selected{% endif %}>0</option>
            <option value="1" {% if form.get(f, '0') == '1' %}selected{% endif %}>1</option>
          </select>
        </div>
        {% endfor %}
      </div>
      <button class="btn" type="submit">Calculate</button>
    </form>

    {% if prob is not none %}
      <div class="result">
        <div><b>Predicted probability (STACK)</b>: {{ prob }}</div>
        <div class="small">Class (threshold=0.5): {{ cls }}</div>
      </div>
    {% endif %}

    {% if force_svg or waterfall_svg %}
      <div class="plots">
        {% if force_svg %}
        <div class="plotbox">
          <h4>SVM SHAP Force Plot (single patient)</h4>
          {{ force_svg|safe }}
        </div>
        {% endif %}
        {% if waterfall_svg %}
        <div class="plotbox">
          <h4>SVM SHAP Waterfall Plot (single patient)</h4>
          {{ waterfall_svg|safe }}
        </div>
        {% endif %}
      </div>
    {% endif %}

    <div class="note">
      <div><b>How to fill binary variables</b></div>
      <ul>
        <li>Thresholded continuous features (1 if value ≥ threshold, else 0): <span class="mono">{{ thresholds }}</span></li>
        <li>Fixed binary features (0/1): <span class="mono">{{ fixed }}</span></li>
      </ul>
      <div class="small">Models: STACK (DT/RF/XGB/SVM/GBDT + LogisticRegression meta, 5-fold CV stacking) and SVM SHAP (KernelExplainer, background n=100).</div>
    </div>
  </div>
</body>
</html>
"""

def _parse_inputs(form_like: Dict[str, Any]) -> Dict[str, int]:
    out = {}
    for f in FEATURE_ORDER:
        v = form_like.get(f, 0)
        try:
            iv = int(v)
        except Exception:
            raise ValueError(f"Invalid value for {f}: {v}")
        if iv not in (0, 1):
            raise ValueError(f"Value for {f} must be 0 or 1, got: {iv}")
        out[f] = iv
    return out

def _predict_stack(x_dict: Dict[str, int]) -> float:
    x = np.array([x_dict[f] for f in FEATURE_ORDER], dtype=float).reshape(1, -1)
    p = float(STACK_MODEL.predict_proba(x)[:, 1][0])
    return p

def _compute_svm_shap_svgs(x_dict: Dict[str, int], nsamples: int = 200) -> Tuple[str, str]:
    """
    Returns (force_svg_text, waterfall_svg_text).
    """
    x = np.array([x_dict[f] for f in FEATURE_ORDER], dtype=float).reshape(1, -1)
    vals_obj = SVM_EXPLAINER.shap_values(x, nsamples=nsamples)

    sv = np.array(vals_obj)
    if isinstance(vals_obj, list):
        sv = np.array(vals_obj[0])
    sv = sv.reshape(-1)

    base_val = float(SVM_EXPLAINER.expected_value) if not isinstance(SVM_EXPLAINER.expected_value, (list, np.ndarray)) else float(np.array(SVM_EXPLAINER.expected_value).ravel()[-1])

    force_path = os.path.join(WEB_OUT_DIR, "svm_force_single.svg")
    wf_path = os.path.join(WEB_OUT_DIR, "svm_waterfall_single.svg")

    # Use your standardized helper to generate SVGs
    shap_force_waterfall_svg(
        shap_values_row=sv,
        X_row=x.reshape(-1),
        feature_names=FEATURE_ORDER,
        base_value=base_val,
        out_force_svg=force_path,
        out_waterfall_svg=wf_path,
    )

    def _read_svg(p: str) -> str:
        if not os.path.exists(p):
            return ""
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    return _read_svg(force_path), _read_svg(wf_path)

@app.route("/", methods=["GET", "POST"])
def index():
    prob = None
    cls = None
    form = {}
    force_svg = ""
    waterfall_svg = ""

    if request.method == "POST":
        form = request.form.to_dict()
        try:
            x_dict = _parse_inputs(form)
            p = _predict_stack(x_dict)
            prob = f"{p:.4f}"
            cls = int(p >= 0.5)

            # SVM SHAP plots for this patient input
            force_svg, waterfall_svg = _compute_svm_shap_svgs(x_dict, nsamples=200)
        except Exception as e:
            prob = f"ERROR: {e}"
            cls = None

    return render_template_string(
        HTML,
        title=APP_TITLE,
        features=FEATURE_ORDER,
        display_names=DISPLAY_NAMES,
        thresholds=json.dumps(THRESHOLDS, ensure_ascii=False),
        fixed=json.dumps(FIXED, ensure_ascii=False),
        prob=prob,
        cls=cls,
        form=form,
        force_svg=force_svg,
        waterfall_svg=waterfall_svg,
    )

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True, silent=False)
    if not isinstance(data, dict):
        return jsonify({"error": "JSON body must be an object/dict"}), 400
    try:
        x_dict = _parse_inputs(data)
        p = _predict_stack(x_dict)
        force_svg, waterfall_svg = _compute_svm_shap_svgs(x_dict, nsamples=200)
        return jsonify({
            "probability": p,
            "threshold": 0.5,
            "class": int(p >= 0.5),
            "feature_order": FEATURE_ORDER,
            "svm_shap_force_svg": force_svg,
            "svm_shap_waterfall_svg": waterfall_svg,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    log(f"Starting web calculator on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)