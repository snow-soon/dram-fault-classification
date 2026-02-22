"""
Track B: Anomaly Detection
- Models: IsolationForest vs MLP Autoencoder (sklearn)
- 학습: 정상(Normal) 데이터만으로 학습
- 평가: 불량 탐지율(Recall), 오탐율(FPR), ROC-AUC, PR-AUC
- 추가: 불량 유형별 탐지율 분석 (어떤 불량이 탐지 어려운지)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (roc_auc_score, precision_recall_curve,
                             average_precision_score, confusion_matrix)

# ── 데이터 로드 ───────────────────────────────────────────────────────────────
print("=" * 60)
print("Track B: Anomaly Detection")
print("=" * 60)

df = pd.read_csv("data/dram_fault_v2.csv")
FEATURES = ["latency_ns","error_rate_pct","temperature_c","voltage_v",
            "refresh_rate_ms","power_consumption_w","age_days","bandwidth_gbps"]
CLASS_NAMES = {0:"Normal", 1:"Cell_Failure", 2:"Retention_Failure",
               3:"Bridge_Defect", 4:"Open_Circuit"}

# Anomaly Detection 설정: 정상=0, 이상=1
df["anomaly_label"] = (df["label"] != 0).astype(int)

# 학습용: 정상 데이터만 (80%)
normal_df = df[df["label"] == 0].sample(frac=1, random_state=42)
train_normal = normal_df.iloc[:int(len(normal_df) * 0.8)]

# 테스트: 정상 20% + 모든 불량
test_normal  = normal_df.iloc[int(len(normal_df) * 0.8):]
test_fault   = df[df["label"] != 0]
test_df      = pd.concat([test_normal, test_fault]).sample(frac=1, random_state=0)

X_train = train_normal[FEATURES].values
X_test  = test_df[FEATURES].values
y_test  = test_df["anomaly_label"].values
y_test_cls = test_df["label"].values

print(f"\nTrain (Normal only): {len(X_train):,}")
print(f"Test (Normal + All Faults):")
print(f"  Normal: {(y_test==0).sum():,}")
print(f"  Fault:  {(y_test==1).sum():,}")

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

results = {}

# ────────────────────────────────────────────────────────────────────────────
# Model 1: Isolation Forest
# ────────────────────────────────────────────────────────────────────────────
print("\n" + "─"*50)
print("Training Isolation Forest...")

# contamination: 실제 불량률 반영 (~7%)
iso = IsolationForest(n_estimators=200, contamination=0.07,
                      max_samples=0.8, random_state=42, n_jobs=-1)
iso.fit(X_train_s)

# score: 낮을수록 이상 → 부호 반전해서 높을수록 이상으로 변환
iso_score  = -iso.score_samples(X_test_s)
iso_pred   = (iso.predict(X_test_s) == -1).astype(int)  # -1=anomaly

iso_auc    = roc_auc_score(y_test, iso_score)
iso_ap     = average_precision_score(y_test, iso_score)
iso_recall = (iso_pred[y_test == 1] == 1).mean()
iso_fpr    = (iso_pred[y_test == 0] == 1).mean()

print(f"ROC-AUC:  {iso_auc:.4f}")
print(f"PR-AUC:   {iso_ap:.4f}")
print(f"Fault Recall (탐지율): {iso_recall:.4f}")
print(f"FPR (오탐율):          {iso_fpr:.4f}")

results["Isolation_Forest"] = {
    "score": iso_score, "pred": iso_pred,
    "auc": iso_auc, "ap": iso_ap,
    "recall": iso_recall, "fpr": iso_fpr,
}

# ────────────────────────────────────────────────────────────────────────────
# Model 2: MLP Autoencoder (정상 패턴 학습 → 재구성 오류로 이상 탐지)
# ────────────────────────────────────────────────────────────────────────────
print("\n" + "─"*50)
print("Training MLP Autoencoder...")
print("  (Encoder: 8→16→8→4, Decoder: 4→8→16→8)")

ae = MLPRegressor(
    hidden_layer_sizes=(16, 8, 4, 8, 16),
    activation="relu",
    max_iter=200,
    learning_rate_init=0.001,
    random_state=42,
    verbose=False,
)
ae.fit(X_train_s, X_train_s)

# 재구성 오류 (MSE per sample)
X_test_recon  = ae.predict(X_test_s)
recon_error   = np.mean((X_test_s - X_test_recon) ** 2, axis=1)

# 임계값: 정상 재구성 오류 95 percentile
X_train_recon = ae.predict(X_train_s)
train_error   = np.mean((X_train_s - X_train_recon) ** 2, axis=1)
threshold     = np.percentile(train_error, 95)
print(f"  Anomaly threshold (95th pctile of train error): {threshold:.6f}")

ae_pred  = (recon_error > threshold).astype(int)
ae_auc   = roc_auc_score(y_test, recon_error)
ae_ap    = average_precision_score(y_test, recon_error)
ae_recall = (ae_pred[y_test == 1] == 1).mean()
ae_fpr    = (ae_pred[y_test == 0] == 1).mean()

print(f"ROC-AUC:  {ae_auc:.4f}")
print(f"PR-AUC:   {ae_ap:.4f}")
print(f"Fault Recall (탐지율): {ae_recall:.4f}")
print(f"FPR (오탐율):          {ae_fpr:.4f}")

results["Autoencoder"] = {
    "score": recon_error, "pred": ae_pred,
    "auc": ae_auc, "ap": ae_ap,
    "recall": ae_recall, "fpr": ae_fpr,
}

# ── 불량 유형별 탐지율 분석 ───────────────────────────────────────────────────
print("\n" + "─"*50)
print("Per-fault-type detection rate:")
per_type = {}
for model_name, res in results.items():
    per_type[model_name] = {}
    for cls in range(1, 5):
        mask = y_test_cls == cls
        if mask.sum() > 0:
            det_rate = (res["pred"][mask] == 1).mean()
            per_type[model_name][CLASS_NAMES[cls]] = round(float(det_rate), 4)
            print(f"  {model_name} / {CLASS_NAMES[cls]}: {det_rate:.3f}")

# ── 시각화 1: ROC Curve 비교 ──────────────────────────────────────────────────
from sklearn.metrics import roc_curve
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

colors = {"Isolation_Forest": "#1976D2", "Autoencoder": "#E53935"}
for name, res in results.items():
    fpr_c, tpr_c, _ = roc_curve(y_test, res["score"])
    axes[0].plot(fpr_c, tpr_c, color=colors[name],
                 label=f"{name} (AUC={res['auc']:.3f})", lw=2)
axes[0].plot([0,1],[0,1],"k--",alpha=0.4)
axes[0].set_xlabel("False Positive Rate", fontsize=10)
axes[0].set_ylabel("True Positive Rate", fontsize=10)
axes[0].set_title("ROC Curve — Anomaly Detection", fontsize=11)
axes[0].legend(fontsize=9)

# PR Curve
for name, res in results.items():
    prec, rec, _ = precision_recall_curve(y_test, res["score"])
    axes[1].plot(rec, prec, color=colors[name],
                 label=f"{name} (AP={res['ap']:.3f})", lw=2)
baseline = y_test.mean()
axes[1].axhline(baseline, color="gray", linestyle="--", alpha=0.5,
                label=f"Baseline (prevalence={baseline:.3f})")
axes[1].set_xlabel("Recall", fontsize=10)
axes[1].set_ylabel("Precision", fontsize=10)
axes[1].set_title("Precision-Recall Curve\n(more informative for imbalanced data)", fontsize=11)
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig("reports/figures/v2_anomaly_roc_pr.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n✅ Saved ROC/PR curves")

# ── 시각화 2: 불량 유형별 탐지율 ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
fault_names = [CLASS_NAMES[i] for i in range(1,5)]
x = np.arange(len(fault_names))
width = 0.35
model_colors = ["#1976D2", "#E53935"]

for i, (name, type_dict) in enumerate(per_type.items()):
    vals = [type_dict.get(fn, 0) for fn in fault_names]
    bars = ax.bar(x + i * width, vals, width, label=name,
                  color=model_colors[i], alpha=0.85)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{v:.2f}", ha="center", fontsize=9)

ax.set_xticks(x + width/2)
ax.set_xticklabels(fault_names, rotation=15, ha="right", fontsize=9)
ax.set_ylabel("Detection Rate (Recall)", fontsize=10)
ax.set_title("Per-Fault-Type Detection Rate\nAnomaly Detection Models", fontsize=11)
ax.legend(fontsize=9)
ax.set_ylim(0, 1.15)
ax.axhline(0.8, color="gray", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("reports/figures/v2_anomaly_per_type.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Saved per-type detection rate")

# ── 시각화 3: Reconstruction Error Distribution (Autoencoder) ────────────────
fig, ax = plt.subplots(figsize=(9, 5))
colors_cls = ["#4CAF50","#F44336","#FF9800","#2196F3","#9C27B0"]

# 클래스별 재구성 오류 분포
for cls in range(5):
    mask = y_test_cls == cls
    if mask.sum() > 0:
        ax.hist(recon_error[mask], bins=60, alpha=0.6, color=colors_cls[cls],
                label=CLASS_NAMES[cls], density=True)

ax.axvline(threshold, color="black", linestyle="--", lw=2,
           label=f"Threshold={threshold:.4f}")
ax.set_xlabel("Reconstruction Error (MSE)", fontsize=10)
ax.set_ylabel("Density", fontsize=10)
ax.set_title("Autoencoder Reconstruction Error by Class\n"
             "(Good separation = higher threshold → lower FPR)", fontsize=11)
ax.legend(fontsize=8)
ax.set_xlim(left=0)
plt.tight_layout()
plt.savefig("reports/figures/v2_autoencoder_recon_error.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Saved reconstruction error distribution")

# ── 결과 저장 ─────────────────────────────────────────────────────────────────
summary_b = {}
for name, res in results.items():
    summary_b[name] = {
        "roc_auc": round(float(res["auc"]), 4),
        "pr_auc":  round(float(res["ap"]),  4),
        "fault_recall": round(float(res["recall"]), 4),
        "false_positive_rate": round(float(res["fpr"]), 4),
        "per_fault_detection": per_type[name],
    }

with open("reports/track_b_results.json", "w") as f:
    json.dump(summary_b, f, indent=2)

print("\n✅ Saved: reports/track_b_results.json")
print("\n🎉 Track B complete!")
print(json.dumps(summary_b, indent=2))
