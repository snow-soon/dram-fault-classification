"""
DRAM Fault Pattern Dataset Generation (v2)
- 5 classes: Normal, Cell Failure, Retention Failure, Bridge Defect, Open Circuit
- Realistic class imbalance: Normal 93%, Faults 7%
- Borderline cases & measurement noise included
- 100,000 samples
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

np.random.seed(42)
N_TOTAL = 100_000

# ── 클래스 비율 (실제 공정 불량률 반영) ──────────────────────────────────────
CLASS_RATIOS = {
    0: 0.93,   # Normal
    1: 0.030,  # Cell Failure
    2: 0.015,  # Retention Failure
    3: 0.015,  # Bridge Defect
    4: 0.010,  # Open Circuit
}
CLASS_NAMES = {
    0: "Normal",
    1: "Cell_Failure",
    2: "Retention_Failure",
    3: "Bridge_Defect",
    4: "Open_Circuit",
}

# ── 피처 정의 (단위 포함) ─────────────────────────────────────────────────────
FEATURES = [
    "latency_ns",        # 메모리 접근 지연
    "error_rate_pct",    # 비트 에러율
    "temperature_c",     # 동작 온도
    "voltage_v",         # 공급 전압
    "refresh_rate_ms",   # 리프레시 주기
    "power_consumption_w",  # 소비 전력
    "age_days",          # 사용 기간
    "bandwidth_gbps",    # 메모리 대역폭
]

# ── 각 클래스별 피처 분포 파라미터 ───────────────────────────────────────────
# 형식: (mean, std) — Normal 기준값 대비 불량 유형별 특성적 편차를 설계
# Normal 정상 범위:
#   latency: 15±2 ns
#   error_rate: 0.0005±0.0002 %
#   temperature: 50±8 °C
#   voltage: 1.20±0.02 V
#   refresh_rate: 64±2 ms
#   power: 2.0±0.3 W
#   age: 500±400 days
#   bandwidth: 28±2 Gbps

def sample_class(n, cls):
    """클래스별 피처 샘플링. 각 불량 유형은 Normal과 일부 겹치는 분포를 가짐."""
    
    if cls == 0:  # Normal
        latency       = np.random.normal(15.0,  2.0, n).clip(10, 22)
        error_rate    = np.random.exponential(0.0004, n).clip(0, 0.002)
        temperature   = np.random.normal(50.0,  8.0, n).clip(30, 70)
        voltage       = np.random.normal(1.200, 0.020, n).clip(1.15, 1.25)
        refresh_rate  = np.random.normal(64.0,  2.0, n).clip(58, 70)
        power         = np.random.normal(2.00,  0.30, n).clip(1.2, 2.8)
        age           = np.random.exponential(400, n).clip(10, 1500)
        bandwidth     = np.random.normal(28.0,  2.0, n).clip(23, 32)

    elif cls == 1:  # Cell Failure: error_rate 급증, latency 약간 증가
        # borderline: 하위 30%는 Normal 범위와 겹치도록
        latency       = np.random.normal(22.0,  8.0, n).clip(12, 55)   # 겹침 있음
        error_rate    = np.random.exponential(0.08, n).clip(0.001, 0.5) # 핵심 시그널
        temperature   = np.random.normal(55.0,  9.0, n).clip(35, 78)
        voltage       = np.random.normal(1.195, 0.025, n).clip(1.14, 1.25)
        refresh_rate  = np.random.normal(62.0,  3.5, n).clip(54, 70)
        power         = np.random.normal(2.30,  0.45, n).clip(1.5, 3.5)
        age           = np.random.normal(1200,  500, n).clip(100, 2500)
        bandwidth     = np.random.normal(25.5,  3.0, n).clip(16, 32)  # 겹침 있음

    elif cls == 2:  # Retention Failure: refresh_rate 이상, temperature 높음
        latency       = np.random.normal(18.0,  4.5, n).clip(11, 38)  # 겹침 있음
        error_rate    = np.random.exponential(0.02, n).clip(0.0005, 0.15) # 중간 수준
        temperature   = np.random.normal(68.0,  8.0, n).clip(48, 85)  # 핵심
        voltage       = np.random.normal(1.190, 0.030, n).clip(1.13, 1.26)
        refresh_rate  = np.random.normal(56.0,  5.0, n).clip(42, 68)  # 핵심: 주기 짧아짐
        power         = np.random.normal(2.50,  0.50, n).clip(1.6, 4.0)
        age           = np.random.normal(1500,  600, n).clip(200, 3000)
        bandwidth     = np.random.normal(26.0,  2.5, n).clip(18, 32)  # 겹침 있음

    elif cls == 3:  # Bridge Defect: voltage 불안정, bandwidth 감소, 노이즈 큼
        latency       = np.random.normal(20.0,  6.0, n).clip(12, 45)
        error_rate    = np.random.exponential(0.015, n).clip(0.0003, 0.1)
        temperature   = np.random.normal(53.0,  10.0, n).clip(33, 75) # 겹침 있음
        # voltage 불안정: 양봉 분포 (정상 근처 + 저전압 쪽)
        v_mode = np.random.choice([0, 1], n, p=[0.4, 0.6])
        voltage = np.where(v_mode == 0,
                           np.random.normal(1.185, 0.035, n),
                           np.random.normal(1.145, 0.025, n)).clip(1.10, 1.25)  # 핵심: 저전압
        refresh_rate  = np.random.normal(62.5,  4.0, n).clip(52, 72)
        power         = np.random.normal(2.20,  0.55, n).clip(1.3, 3.8)
        age           = np.random.normal(900,   500, n).clip(50, 2200)
        bandwidth     = np.random.normal(23.5,  3.5, n).clip(14, 30)  # 핵심: 감소

    elif cls == 4:  # Open Circuit: latency 급증, bandwidth 급감, power 증가
        latency       = np.random.normal(45.0,  18.0, n).clip(20, 100)  # 핵심
        error_rate    = np.random.exponential(0.04, n).clip(0.001, 0.3)
        temperature   = np.random.normal(60.0,  9.0, n).clip(38, 82)
        voltage       = np.random.normal(1.185, 0.030, n).clip(1.12, 1.25)
        refresh_rate  = np.random.normal(61.0,  4.5, n).clip(50, 72)
        power         = np.random.normal(3.20,  0.70, n).clip(1.8, 5.5)  # 핵심
        age           = np.random.normal(1800,  700, n).clip(300, 3500)
        bandwidth     = np.random.normal(19.0,  4.0, n).clip(10, 28)  # 핵심

    df = pd.DataFrame({
        "latency_ns":          latency,
        "error_rate_pct":      error_rate,
        "temperature_c":       temperature,
        "voltage_v":           voltage,
        "refresh_rate_ms":     refresh_rate,
        "power_consumption_w": power,
        "age_days":            age,
        "bandwidth_gbps":      bandwidth,
    })
    return df


# ── 측정 노이즈 추가 (센서 오차 시뮬레이션) ───────────────────────────────────
NOISE_STD = {
    "latency_ns":          0.3,
    "error_rate_pct":      0.00005,
    "temperature_c":       0.5,
    "voltage_v":           0.003,
    "refresh_rate_ms":     0.2,
    "power_consumption_w": 0.05,
    "age_days":            0.0,    # 노이즈 없음
    "bandwidth_gbps":      0.2,
}


def add_measurement_noise(df):
    df = df.copy()
    for col, std in NOISE_STD.items():
        if std > 0:
            df[col] += np.random.normal(0, std, len(df))
    return df


# ── 데이터 생성 ───────────────────────────────────────────────────────────────
print("Generating dataset...")
dfs = []
for cls, ratio in CLASS_RATIOS.items():
    n = int(N_TOTAL * ratio)
    df_cls = sample_class(n, cls)
    df_cls = add_measurement_noise(df_cls)
    df_cls["label"] = cls
    df_cls["fault_type"] = CLASS_NAMES[cls]
    dfs.append(df_cls)
    print(f"  Class {cls} ({CLASS_NAMES[cls]}): {n:,} samples")

df_all = pd.concat(dfs, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"\nTotal: {len(df_all):,} samples")
print("\nLabel distribution:")
print(df_all["label"].value_counts().sort_index())

# ── 저장 ─────────────────────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
df_all.to_csv("data/dram_fault_v2.csv", index=False)
print("\n✅ Saved: data/dram_fault_v2.csv")

# ── 시각화 1: 클래스 분포 ─────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
colors = ["#4CAF50", "#F44336", "#FF9800", "#2196F3", "#9C27B0"]
labels_sorted = [CLASS_NAMES[i] for i in range(5)]

for idx, feat in enumerate(FEATURES):
    ax = axes[idx]
    for cls in range(5):
        subset = df_all[df_all["label"] == cls][feat]
        ax.hist(subset, bins=50, alpha=0.5, color=colors[cls],
                label=CLASS_NAMES[cls], density=True)
    ax.set_title(feat, fontsize=10)
    ax.set_xlabel("")
    ax.tick_params(labelsize=8)

axes[0].legend(fontsize=7, loc="upper right")
# 빈 서브플롯 제거
for i in range(len(FEATURES), len(axes)):
    fig.delaxes(axes[i])

plt.suptitle("Feature Distributions by Fault Type\n(overlapping regions = borderline cases)",
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("reports/figures/v2_feature_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Saved: reports/figures/v2_feature_distributions.png")

# ── 시각화 2: 클래스 불균형 파이차트 ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

counts = df_all["label"].value_counts().sort_index()
names = [CLASS_NAMES[i] for i in counts.index]

# 전체 파이
axes[0].pie(counts, labels=names, colors=colors,
            autopct="%1.1f%%", startangle=90,
            textprops={"fontsize": 9})
axes[0].set_title("Overall Class Distribution\n(93% Normal, 7% Faults)", fontsize=11)

# 불량만 확대
fault_counts = counts[1:]
fault_names = [CLASS_NAMES[i] for i in fault_counts.index]
axes[1].pie(fault_counts, labels=fault_names, colors=colors[1:],
            autopct="%1.1f%%", startangle=90,
            textprops={"fontsize": 9})
axes[1].set_title("Fault Types Only\n(breakdown among defective chips)", fontsize=11)

plt.tight_layout()
plt.savefig("reports/figures/v2_class_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Saved: reports/figures/v2_class_distribution.png")

# ── 시각화 3: 주요 피처 쌍 산점도 (borderline 확인) ──────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
pairs = [
    ("error_rate_pct", "latency_ns"),
    ("temperature_c",  "refresh_rate_ms"),
    ("bandwidth_gbps", "power_consumption_w"),
]
sample = df_all.sample(5000, random_state=0)  # 5천개만 플롯

for ax, (x, y) in zip(axes, pairs):
    for cls in range(5):
        sub = sample[sample["label"] == cls]
        ax.scatter(sub[x], sub[y], c=colors[cls], alpha=0.35, s=8,
                   label=CLASS_NAMES[cls])
    ax.set_xlabel(x, fontsize=9)
    ax.set_ylabel(y, fontsize=9)
    ax.tick_params(labelsize=8)

axes[0].legend(fontsize=7, markerscale=2)
plt.suptitle("Key Feature Pair Scatter Plots — Borderline Overlap Visible", fontsize=12)
plt.tight_layout()
plt.savefig("reports/figures/v2_scatter_borderline.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Saved: reports/figures/v2_scatter_borderline.png")

print("\n🎉 Data generation complete!")
