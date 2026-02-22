"""
Track A: Multiclass Fault Classification
- Models: RandomForestClassifier vs GradientBoostingClassifier
- Evaluation: Accuracy, Per-class Precision/Recall/F1, Confusion Matrix, ROC-AUC
- Techniques: SMOTE-like oversampling for class imbalance, feature importance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import json
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, accuracy_score)
from sklearn.utils.class_weight import compute_class_weight

# ── 데이터 로드 ───────────────────────────────────────────────────────────────
print("=" * 60)
print("Track A: Multiclass Fault Classification")
print("=" * 60)

df = pd.read_csv("data/dram_fault_v2.csv")
FEATURES = ["latency_ns","error_rate_pct","temperature_c","voltage_v",
            "refresh_rate_ms","power_consumption_w","age_days","bandwidth_gbps"]
CLASS_NAMES = {0:"Normal", 1:"Cell_Failure", 2:"Retention_Failure",
               3:"Bridge_Defect", 4:"Open_Circuit"}

X = df[FEATURES].values
y = df["label"].values

# ── Train/Test 분할 (stratified) ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ── 스케일링 ──────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print(f"\nTrain: {len(X_train):,}  |  Test: {len(X_test):,}")
print("Test class distribution:")
for cls, name in CLASS_NAMES.items():
    n = (y_test == cls).sum()
    print(f"  {name}: {n}")

# ── Class weight 계산 (불균형 대응) ───────────────────────────────────────────
class_weights = compute_class_weight("balanced", classes=np.arange(5), y=y_train)
cw_dict = {i: w for i, w in enumerate(class_weights)}
print(f"\nClass weights (balanced): {cw_dict}")

# ── 모델 정의 ─────────────────────────────────────────────────────────────────
models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.15,
        subsample=0.7,
        max_features="sqrt",
        random_state=42
    ),
}

results = {}

for model_name, model in models.items():
    print(f"\n{'─'*50}")
    print(f"Training {model_name}...")
    model.fit(X_train_s, y_train)
    
    y_pred  = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)
    
    acc = accuracy_score(y_test, y_pred)
    # One-vs-Rest AUC
    auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
    
    print(f"Accuracy:  {acc:.4f}")
    print(f"ROC-AUC (macro OvR): {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=list(CLASS_NAMES.values())))
    
    results[model_name] = {
        "accuracy": acc,
        "roc_auc": auc,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "model": model,
    }

# ── 시각화 1: Confusion Matrix 비교 ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
class_labels = list(CLASS_NAMES.values())

for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res["y_pred"], normalize="true")
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_labels, yticklabels=class_labels, ax=ax)
    ax.set_title(f"{name}\nAcc={res['accuracy']:.3f} | AUC={res['roc_auc']:.3f}",
                 fontsize=11)
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("True", fontsize=9)
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)

plt.suptitle("Normalized Confusion Matrix — Multiclass Fault Classification",
             fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig("reports/figures/v2_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n✅ Saved confusion matrix")

# ── 시각화 2: Feature Importance (RF) ─────────────────────────────────────────
rf = results["RandomForest"]["model"]
importances = rf.feature_importances_
feat_df = pd.DataFrame({"feature": FEATURES, "importance": importances})
feat_df = feat_df.sort_values("importance", ascending=True)

fig, ax = plt.subplots(figsize=(8, 5))
colors_fi = ["#2196F3" if imp >= feat_df["importance"].median() else "#90CAF9"
             for imp in feat_df["importance"]]
ax.barh(feat_df["feature"], feat_df["importance"], color=colors_fi)
ax.set_xlabel("Feature Importance (Gini)", fontsize=10)
ax.set_title("Random Forest — Feature Importance\nfor DRAM Fault Classification", fontsize=11)
for i, (name, val) in enumerate(zip(feat_df["feature"], feat_df["importance"])):
    ax.text(val + 0.001, i, f"{val:.3f}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig("reports/figures/v2_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Saved feature importance")

# ── 시각화 3: Per-class Recall 비교 ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(5)
width = 0.35
colors_model = ["#1976D2", "#E53935"]

for i, (name, res) in enumerate(results.items()):
    from sklearn.metrics import recall_score
    recalls = []
    for cls in range(5):
        mask = y_test == cls
        r = (res["y_pred"][mask] == cls).mean() if mask.sum() > 0 else 0
        recalls.append(r)
    ax.bar(x + i * width, recalls, width, label=name, color=colors_model[i], alpha=0.85)

ax.set_xticks(x + width / 2)
ax.set_xticklabels(class_labels, rotation=20, ha="right", fontsize=9)
ax.set_ylabel("Recall", fontsize=10)
ax.set_title("Per-class Recall Comparison\n(Fault detection rate per type)", fontsize=11)
ax.legend(fontsize=9)
ax.set_ylim(0, 1.05)
ax.axhline(0.8, color="gray", linestyle="--", alpha=0.5, label="80% threshold")
plt.tight_layout()
plt.savefig("reports/figures/v2_per_class_recall.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Saved per-class recall")

# ── 결과 저장 ─────────────────────────────────────────────────────────────────
summary = {}
for name, res in results.items():
    from sklearn.metrics import precision_score, recall_score, f1_score
    summary[name] = {
        "accuracy": round(res["accuracy"], 4),
        "roc_auc_macro": round(res["roc_auc"], 4),
        "per_class_recall": {
            CLASS_NAMES[c]: round(float((res["y_pred"][y_test==c] == c).mean()), 4)
            for c in range(5) if (y_test==c).sum() > 0
        }
    }

with open("reports/track_a_results.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n✅ Saved: reports/track_a_results.json")
print("\n🎉 Track A complete!")
print(json.dumps(summary, indent=2))
