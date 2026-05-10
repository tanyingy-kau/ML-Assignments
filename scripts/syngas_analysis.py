"""
Syngas Fermentation Ethanol Prediction - Full Analysis Script
CIS 732 Term Project - Ying Tan
Generates all figures required for report and presentation.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import ttest_rel, wilcoxon
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_URL = (
    "https://raw.githubusercontent.com/garrettroell/"
    "SyngasMachineLearning/main/data/experimental_data.csv"
)
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

FEATURES = ["time", "N2", "CO", "CO2", "H2", "flow rate (mL/min)", "composition"]
TARGET = "ethanol (mM)"
RANDOM_SEED = 42
N_SPLITS = 30


# ── Helpers ────────────────────────────────────────────────────────────────────
def metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


def savefig(name):
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("Loading data...")
df = pd.read_csv(DATA_URL)
X = df[FEATURES]
y = df[TARGET]
print(f"  Dataset: {len(df)} samples, {len(FEATURES)} features")
print(f"  Target range: {y.min():.2f} – {y.max():.2f} mM")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Data Overview — Feature Distributions
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Fig 1] Data overview: feature distributions...")
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
fig.suptitle("Figure 1: Feature and Target Distributions (n=176)", fontsize=14, fontweight="bold")
all_cols = FEATURES + [TARGET]
for ax, col in zip(axes.flatten(), all_cols):
    ax.hist(df[col], bins=20, color="#4C72B0", edgecolor="white", alpha=0.85)
    ax.set_title(col, fontsize=10)
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
axes.flatten()[-1].set_visible(False)  # hide spare cell
plt.tight_layout()
savefig("fig1_data_distributions.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Data Overview — Target vs. Each Feature (Scatter)
# ══════════════════════════════════════════════════════════════════════════════
print("[Fig 2] Feature vs. target scatter plots...")
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
fig.suptitle("Figure 2: Ethanol (mM) vs. Each Input Feature", fontsize=14, fontweight="bold")
for ax, feat in zip(axes.flatten(), FEATURES):
    ax.scatter(df[feat], y, alpha=0.5, s=15, color="#DD8452")
    ax.set_xlabel(feat, fontsize=9)
    ax.set_ylabel("Ethanol (mM)")
    ax.grid(True, alpha=0.3)
for ax in axes.flatten()[len(FEATURES):]:
    ax.set_visible(False)
plt.tight_layout()
savefig("fig2_feature_vs_target.png")

# ══════════════════════════════════════════════════════════════════════════════
# 2. TRAIN / TEST SPLIT (fixed seed=42)
# ══════════════════════════════════════════════════════════════════════════════
print("\nSplitting data (80/20, seed=42)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED
)
print(f"  Train: {len(X_train)} samples | Test: {len(X_test)} samples")

# ── Scale for MLP ─────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ══════════════════════════════════════════════════════════════════════════════
# 3. TRAIN MODELS
# ══════════════════════════════════════════════════════════════════════════════
print("\nTraining models...")

lr = LinearRegression()
lr.fit(X_train, y_train)
pred_lr_train = lr.predict(X_train)
pred_lr_test = lr.predict(X_test)

rf = RandomForestRegressor(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1)
rf.fit(X_train, y_train)
pred_rf_train = rf.predict(X_train)
pred_rf_test = rf.predict(X_test)

# MLP: record loss curve during training
mlp = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    max_iter=1,
    warm_start=True,
    random_state=RANDOM_SEED,
    learning_rate_init=0.001,
)
train_losses = []
MAX_ITER = 500
for epoch in range(MAX_ITER):
    mlp.fit(X_train_s, y_train)
    train_losses.append(mlp.loss_)

pred_mlp_train = mlp.predict(X_train_s)
pred_mlp_test = mlp.predict(X_test_s)

print("  Linear Regression — done")
print("  Random Forest     — done")
print("  MLP               — done")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: MLP Training Loss Curve
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Fig 3] MLP training loss curve...")
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(1, MAX_ITER + 1), train_losses, color="#55A868", linewidth=1.5)
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Training Loss (MSE)", fontsize=12)
ax.set_title("Figure 3: MLP Training Loss Curve (500 epochs)", fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3)
plt.tight_layout()
savefig("fig3_mlp_training_loss.png")

# ══════════════════════════════════════════════════════════════════════════════
# 4. METRICS TABLE
# ══════════════════════════════════════════════════════════════════════════════
results = {}
for name, y_tr_pred, y_te_pred in [
    ("Linear Regression", pred_lr_train, pred_lr_test),
    ("Random Forest",     pred_rf_train, pred_rf_test),
    ("MLP",              pred_mlp_train, pred_mlp_test),
]:
    tr = metrics(y_train, y_tr_pred)
    te = metrics(y_test,  y_te_pred)
    results[name] = {
        "Train RMSE": tr[0], "Train MAE": tr[1], "Train R²": tr[2],
        "Test RMSE":  te[0], "Test MAE":  te[1], "Test R²":  te[2],
    }
    print(f"  {name:20s}  Train R²={tr[2]:.4f}  Test R²={te[2]:.4f}  "
          f"Test RMSE={te[0]:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: Training vs. Testing Performance Bar Chart (all 3 models)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Fig 4] Train vs. test performance comparison...")
model_names = ["Linear Regression", "Random Forest", "MLP"]
train_r2  = [results[m]["Train R²"]   for m in model_names]
test_r2   = [results[m]["Test R²"]    for m in model_names]
train_rmse = [results[m]["Train RMSE"] for m in model_names]
test_rmse  = [results[m]["Test RMSE"]  for m in model_names]

x = np.arange(len(model_names))
w = 0.35
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Figure 4: Train vs. Test Performance — All Models", fontsize=13, fontweight="bold")

# R²
axes[0].bar(x - w/2, train_r2,  w, label="Train", color="#4C72B0", alpha=0.85)
axes[0].bar(x + w/2, test_r2,   w, label="Test",  color="#DD8452", alpha=0.85)
axes[0].set_xticks(x); axes[0].set_xticklabels(model_names, fontsize=10)
axes[0].set_ylabel("R² Score"); axes[0].set_ylim(0, 1.05)
axes[0].set_title("R² Score (higher = better)")
axes[0].legend(); axes[0].grid(True, alpha=0.3, axis="y")
for i, (tr, te) in enumerate(zip(train_r2, test_r2)):
    axes[0].text(i - w/2, tr + 0.01, f"{tr:.3f}", ha="center", fontsize=8)
    axes[0].text(i + w/2, te + 0.01, f"{te:.3f}", ha="center", fontsize=8)

# RMSE
axes[1].bar(x - w/2, train_rmse, w, label="Train", color="#4C72B0", alpha=0.85)
axes[1].bar(x + w/2, test_rmse,  w, label="Test",  color="#DD8452", alpha=0.85)
axes[1].set_xticks(x); axes[1].set_xticklabels(model_names, fontsize=10)
axes[1].set_ylabel("RMSE (mM)"); axes[1].set_title("RMSE (lower = better)")
axes[1].legend(); axes[1].grid(True, alpha=0.3, axis="y")
for i, (tr, te) in enumerate(zip(train_rmse, test_rmse)):
    axes[1].text(i - w/2, tr + 0.1, f"{tr:.2f}", ha="center", fontsize=8)
    axes[1].text(i + w/2, te + 0.1, f"{te:.2f}", ha="center", fontsize=8)

plt.tight_layout()
savefig("fig4_train_vs_test_performance.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5: Actual vs. Predicted Scatter — LR (Train + Test)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Fig 5] Scatter plots — Linear Regression...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Figure 5: Linear Regression — Actual vs. Predicted", fontsize=13, fontweight="bold")
lims = [y.min() - 5, y.max() + 5]
for ax, y_true, y_pred, split, n in [
    (axes[0], y_train, pred_lr_train, "Training", len(y_train)),
    (axes[1], y_test,  pred_lr_test,  "Testing",  len(y_test)),
]:
    r2_val = r2_score(y_true, y_pred)
    rmse_val = np.sqrt(mean_squared_error(y_true, y_pred))
    ax.scatter(y_true, y_pred, alpha=0.7, s=30, color="#4C72B0", edgecolors="white", linewidths=0.3)
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Actual Ethanol (mM)", fontsize=11)
    ax.set_ylabel("Predicted Ethanol (mM)", fontsize=11)
    ax.set_title(f"{split} Set (n={n})\nR²={r2_val:.4f}, RMSE={rmse_val:.4f} mM", fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout()
savefig("fig5_scatter_LR_train_test.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6: Actual vs. Predicted Scatter — RF (Train + Test)
# ══════════════════════════════════════════════════════════════════════════════
print("[Fig 6] Scatter plots — Random Forest...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Figure 6: Random Forest — Actual vs. Predicted", fontsize=13, fontweight="bold")
for ax, y_true, y_pred, split, n in [
    (axes[0], y_train, pred_rf_train, "Training", len(y_train)),
    (axes[1], y_test,  pred_rf_test,  "Testing",  len(y_test)),
]:
    r2_val = r2_score(y_true, y_pred)
    rmse_val = np.sqrt(mean_squared_error(y_true, y_pred))
    ax.scatter(y_true, y_pred, alpha=0.7, s=30, color="#55A868", edgecolors="white", linewidths=0.3)
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Actual Ethanol (mM)", fontsize=11)
    ax.set_ylabel("Predicted Ethanol (mM)", fontsize=11)
    ax.set_title(f"{split} Set (n={n})\nR²={r2_val:.4f}, RMSE={rmse_val:.4f} mM", fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout()
savefig("fig6_scatter_RF_train_test.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 7: Actual vs. Predicted Scatter — MLP (Train + Test)
# ══════════════════════════════════════════════════════════════════════════════
print("[Fig 7] Scatter plots — MLP...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Figure 7: MLP — Actual vs. Predicted", fontsize=13, fontweight="bold")
for ax, y_true, y_pred, split, n in [
    (axes[0], y_train, pred_mlp_train, "Training", len(y_train)),
    (axes[1], y_test,  pred_mlp_test,  "Testing",  len(y_test)),
]:
    r2_val = r2_score(y_true, y_pred)
    rmse_val = np.sqrt(mean_squared_error(y_true, y_pred))
    ax.scatter(y_true, y_pred, alpha=0.7, s=30, color="#C44E52", edgecolors="white", linewidths=0.3)
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Actual Ethanol (mM)", fontsize=11)
    ax.set_ylabel("Predicted Ethanol (mM)", fontsize=11)
    ax.set_title(f"{split} Set (n={n})\nR²={r2_val:.4f}, RMSE={rmse_val:.4f} mM", fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout()
savefig("fig7_scatter_MLP_train_test.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 8: Feature Importance (Random Forest)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Fig 8] Random Forest feature importance...")
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
feat_sorted = [FEATURES[i] for i in indices]
imp_sorted = importances[indices]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(feat_sorted[::-1], imp_sorted[::-1], color="#4C72B0", alpha=0.85, edgecolor="white")
ax.set_xlabel("Feature Importance (Mean Decrease in Impurity)", fontsize=11)
ax.set_title("Figure 8: Random Forest Feature Importance", fontsize=13, fontweight="bold")
for bar, val in zip(bars, imp_sorted[::-1]):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=9)
ax.grid(True, alpha=0.3, axis="x")
plt.tight_layout()
savefig("fig8_RF_feature_importance.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 9: Statistical Testing — Residual Comparison (LR vs RF vs MLP)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Fig 9] Residual / error comparison...")
err_lr  = np.abs(y_test.values - pred_lr_test)
err_rf  = np.abs(y_test.values - pred_rf_test)
err_mlp = np.abs(y_test.values - pred_mlp_test)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Figure 9: Absolute Error Distribution on Test Set", fontsize=13, fontweight="bold")

# Boxplot
bp = axes[0].boxplot(
    [err_lr, err_mlp, err_rf],
    labels=["LR", "MLP", "RF"],
    patch_artist=True,
    medianprops=dict(color="black", linewidth=2),
)
colors = ["#4C72B0", "#C44E52", "#55A868"]
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color); patch.set_alpha(0.75)
axes[0].set_ylabel("Absolute Error (mM)")
axes[0].set_title("Absolute Error Boxplot (Test Set)")
axes[0].grid(True, alpha=0.3, axis="y")

# Scatter: LR error vs RF error (paired, same test samples)
axes[1].scatter(err_lr, err_rf, alpha=0.7, s=30, color="#4C72B0")
max_err = max(err_lr.max(), err_rf.max()) + 2
axes[1].plot([0, max_err], [0, max_err], "r--", linewidth=1.5, label="Equal error line")
axes[1].set_xlabel("LR Absolute Error (mM)"); axes[1].set_ylabel("RF Absolute Error (mM)")
axes[1].set_title("Paired Error: LR vs RF (points below line = RF better)")
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
savefig("fig9_error_comparison.png")

# ══════════════════════════════════════════════════════════════════════════════
# 5. STATISTICAL TESTS
# ══════════════════════════════════════════════════════════════════════════════
print("\nRunning statistical tests...")
t_stat, t_p   = ttest_rel(err_lr, err_rf, alternative="greater")
w_stat, w_p   = wilcoxon(err_lr, err_rf, alternative="greater")
print(f"  Paired t-test:   t={t_stat:.4f}, p={t_p:.2e}")
print(f"  Wilcoxon test:   W={w_stat:.4f}, p={w_p:.2e}")

# ══════════════════════════════════════════════════════════════════════════════
# 6. 30-FOLD ROBUSTNESS ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nRunning {N_SPLITS}-fold robustness analysis...")
rows = []
for s in range(N_SPLITS):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=s)

    p_lr  = LinearRegression().fit(X_tr, y_tr).predict(X_te)
    p_rf  = RandomForestRegressor(n_estimators=200, random_state=s, n_jobs=-1).fit(X_tr, y_tr).predict(X_te)

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)
    p_mlp = MLPRegressor(
        hidden_layer_sizes=(64, 32), max_iter=2000, random_state=s
    ).fit(X_tr_s, y_tr).predict(X_te_s)

    for name, pred in (("LR", p_lr), ("RF", p_rf), ("MLP", p_mlp)):
        rmse, mae, r2 = metrics(y_te, pred)
        rows.append([s, name, rmse, mae, r2])
    if s % 10 == 9:
        print(f"  Completed seed {s+1}/{N_SPLITS}")

rep = pd.DataFrame(rows, columns=["seed", "model", "rmse", "mae", "r2"])
summary = rep.groupby("model")[["rmse", "mae", "r2"]].agg(["mean", "std"])
print("\n30-Fold Summary:")
print(summary.to_string())

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 10: Robustness — RMSE Distribution Across 30 Splits (Boxplot)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Fig 10] Robustness boxplot (30 splits)...")
lr_rmse  = rep[rep["model"] == "LR"]["rmse"].values
rf_rmse  = rep[rep["model"] == "RF"]["rmse"].values
mlp_rmse = rep[rep["model"] == "MLP"]["rmse"].values

fig, ax = plt.subplots(figsize=(8, 5))
bp = ax.boxplot(
    [lr_rmse, mlp_rmse, rf_rmse],
    labels=["Linear Regression", "MLP", "Random Forest"],
    patch_artist=True,
    medianprops=dict(color="black", linewidth=2),
    notch=False,
)
for patch, color in zip(bp["boxes"], ["#4C72B0", "#C44E52", "#55A868"]):
    patch.set_facecolor(color); patch.set_alpha(0.75)
ax.set_ylabel("RMSE (mM)", fontsize=12)
ax.set_title(f"Figure 10: RMSE Distribution Across {N_SPLITS} Random Splits\n"
             f"LR: {lr_rmse.mean():.2f}±{lr_rmse.std():.2f} | "
             f"MLP: {mlp_rmse.mean():.2f}±{mlp_rmse.std():.2f} | "
             f"RF: {rf_rmse.mean():.2f}±{rf_rmse.std():.2f} mM",
             fontsize=11, fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
savefig("fig10_robustness_boxplot.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 11: Robustness — RMSE Across 30 Seeds (Line Plot)
# ══════════════════════════════════════════════════════════════════════════════
print("[Fig 11] Robustness line plot across seeds...")
fig, ax = plt.subplots(figsize=(10, 5))
seeds = list(range(N_SPLITS))
for model, color, label in [("LR", "#4C72B0", "Linear Regression"),
                              ("RF", "#55A868", "Random Forest"),
                              ("MLP", "#C44E52", "MLP")]:
    rmse_vals = rep[rep["model"] == model].sort_values("seed")["rmse"].values
    ax.plot(seeds, rmse_vals, marker="o", markersize=4, linewidth=1.5,
            color=color, label=label, alpha=0.85)
ax.set_xlabel("Random Seed", fontsize=12)
ax.set_ylabel("RMSE (mM)", fontsize=12)
ax.set_title(f"Figure 11: Test RMSE Across {N_SPLITS} Random Splits", fontsize=13, fontweight="bold")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
savefig("fig11_robustness_line.png")

# ══════════════════════════════════════════════════════════════════════════════
# 7. SEED-LEVEL STATISTICAL TESTS
# ══════════════════════════════════════════════════════════════════════════════
t2_stat, t2_p = ttest_rel(lr_rmse, rf_rmse, alternative="greater")
w2_stat, w2_p = wilcoxon(lr_rmse, rf_rmse, alternative="greater")
print(f"\nSeed-level tests (LR RMSE > RF RMSE):")
print(f"  Paired t-test:  t={t2_stat:.4f}, p={t2_p:.2e}")
print(f"  Wilcoxon test:  W={w2_stat:.4f}, p={w2_p:.2e}")

# ══════════════════════════════════════════════════════════════════════════════
# 8. SAVE RESULTS SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
summary_path = os.path.join(FIG_DIR, "..", "results_summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("=" * 60 + "\n")
    f.write("SYNGAS FERMENTATION — RESULTS SUMMARY\n")
    f.write("CIS 732 Term Project — Ying Tan\n")
    f.write("=" * 60 + "\n\n")

    f.write("── Fixed Split (seed=42, 80/20) ──\n")
    for name in ["Linear Regression", "Random Forest", "MLP"]:
        r = results[name]
        f.write(f"\n{name}:\n")
        f.write(f"  Train: RMSE={r['Train RMSE']:.4f}, MAE={r['Train MAE']:.4f}, R²={r['Train R²']:.4f}\n")
        f.write(f"  Test:  RMSE={r['Test RMSE']:.4f},  MAE={r['Test MAE']:.4f},  R²={r['Test R²']:.4f}\n")

    f.write("\n── Statistical Tests (sample-level, LR vs RF) ──\n")
    f.write(f"  Paired t-test: t={t_stat:.4f}, p={t_p:.2e} {'***SIGNIFICANT' if t_p < 0.05 else ''}\n")
    f.write(f"  Wilcoxon:      W={w_stat:.4f}, p={w_p:.2e} {'***SIGNIFICANT' if w_p < 0.05 else ''}\n")

    f.write("\n── 30-Fold Robustness (RMSE mean ± std) ──\n")
    f.write(f"  LR:  {lr_rmse.mean():.4f} ± {lr_rmse.std():.4f} mM\n")
    f.write(f"  RF:  {rf_rmse.mean():.4f} ± {rf_rmse.std():.4f} mM\n")
    f.write(f"  MLP: {mlp_rmse.mean():.4f} ± {mlp_rmse.std():.4f} mM\n")

    f.write("\n── Seed-level Tests (LR RMSE > RF RMSE) ──\n")
    f.write(f"  Paired t-test: t={t2_stat:.4f}, p={t2_p:.2e} {'***SIGNIFICANT' if t2_p < 0.05 else ''}\n")
    f.write(f"  Wilcoxon:      W={w2_stat:.4f}, p={w2_p:.2e} {'***SIGNIFICANT' if w2_p < 0.05 else ''}\n")

print(f"\n  Results summary saved to: {summary_path}")
print("\n✅ All figures generated successfully!")
print(f"   Location: {os.path.abspath(FIG_DIR)}")
