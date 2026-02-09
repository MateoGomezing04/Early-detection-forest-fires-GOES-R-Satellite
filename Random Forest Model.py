# ============================================================
# RANDOM FOREST WITH 95% CONFIDENCE INTERVALS (PERCENTILES)
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
    confusion_matrix, roc_curve, precision_recall_curve,
    ConfusionMatrixDisplay
)

# ============================================================
# CONFIGURATION
# ============================================================

FILE_NAME = r"C:\Users\User\Desktop\Articulo\Consolidado manual datos dia antes y dia del incendio 12h.xlsx"
SHEET_NAME = "Sheet1"

OUTPUT_DIR = r"C:\Users\User\Desktop\Articulo\RF_Results_CI_Final"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_ITERATIONS = 100
TEST_SIZE = 0.30
RANDOM_SEED_BASE = 100
MIN_SAMPLES = 30

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_excel(FILE_NAME, sheet_name=SHEET_NAME)
df.columns = df.columns.str.strip()

product_col = next(c for c in df.columns if c.lower() == "producto")
value_col = next(c for c in df.columns if c.lower() == "valor")
target_col = next(c for c in df.columns if c.lower() == "incendio")

x_col = next((c for c in df.columns if c.lower() in ["x", "x_grid", "lon"]), None)
y_col = next((c for c in df.columns if c.lower() in ["y", "y_grid", "lat"]), None)

products = df[product_col].unique()
global_summary = []

# ============================================================
# LOOP BY PRODUCT
# ============================================================

for product in products:

    subset = df[df[product_col] == product].copy()
    n_samples = len(subset)

    if n_samples < MIN_SAMPLES or subset[target_col].nunique() < 2:
        continue

    predictors = [value_col]
    if x_col and y_col:
        predictors.extend([x_col, y_col])

    X = subset[predictors].fillna(subset[predictors].mean())
    y = subset[target_col].astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    metrics_iterations = []

    all_y_true = []
    all_y_pred = []
    all_y_prob = []

    # ========================================================
    # MONTE CARLO ITERATIONS
    # ========================================================

    for i in range(N_ITERATIONS):

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled,
            y,
            test_size=TEST_SIZE,
            stratify=y,
            random_state=RANDOM_SEED_BASE + i
        )

        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_SEED_BASE + i,
            n_jobs=-1
        )

        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_test)
        y_prob = rf.predict_proba(X_test)[:, 1]

        all_y_true.append(y_test)
        all_y_pred.append(y_pred)
        all_y_prob.append(y_prob)

        metrics_iterations.append({
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "auc": roc_auc_score(y_test, y_prob)
        })

    metrics_df = pd.DataFrame(metrics_iterations)

    # ========================================================
    # CONFIDENCE INTERVALS
    # ========================================================

    summary = {
        "product": product,
        "n_samples": n_samples
    }

    for metric in metrics_df.columns:
        summary[f"{metric}_mean"] = metrics_df[metric].mean()
        summary[f"{metric}_p2_5"] = np.percentile(metrics_df[metric], 2.5)
        summary[f"{metric}_p97_5"] = np.percentile(metrics_df[metric], 97.5)

    global_summary.append(summary)

    # ========================================================
    # OUTPUT DIRECTORIES
    # ========================================================

    product_dir = os.path.join(OUTPUT_DIR, str(product).replace(" ", "_"))
    os.makedirs(product_dir, exist_ok=True)

    plots_dir = os.path.join(product_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    metrics_df.to_csv(
        os.path.join(product_dir, "metrics_iterations.csv"),
        index=False
    )

    # ========================================================
    # CONFUSION MATRIX (GLOBAL)
    # ========================================================

    y_true_all = np.concatenate(all_y_true)
    y_pred_all = np.concatenate(all_y_pred)

    cm = confusion_matrix(y_true_all, y_pred_all)

    plt.figure(figsize=(4, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", colorbar=True)
    plt.title(f"Confusion Matrix – {product}")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "confusion_matrix.png"), dpi=200)
    plt.close()

    # ========================================================
    # ROC CURVE (MEAN)
    # ========================================================

    mean_fpr = np.linspace(0, 1, 100)
    tprs = []

    for yt, yp in zip(all_y_true, all_y_prob):
        fpr, tpr, _ = roc_curve(yt, yp)
        tprs.append(np.interp(mean_fpr, fpr, tpr))

    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = metrics_df["auc"].mean()

    plt.figure(figsize=(4, 4))
    plt.plot(mean_fpr, mean_tpr, label=f"Mean AUC = {mean_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve – {product}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "roc_curve.png"), dpi=200)
    plt.close()

    # ========================================================
    # PRECISION–RECALL CURVE (MEAN)
    # ========================================================

    recalls = np.linspace(0, 1, 100)
    precisions = []

    for yt, yp in zip(all_y_true, all_y_prob):
        p, r, _ = precision_recall_curve(yt, yp)
        precisions.append(np.interp(recalls, r[::-1], p[::-1]))

    mean_precision = np.mean(precisions, axis=0)

    plt.figure(figsize=(4, 4))
    plt.plot(recalls, mean_precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision–Recall Curve – {product}")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "precision_recall.png"), dpi=200)
    plt.close()

# ============================================================
# SAVE GLOBAL SUMMARY
# ============================================================

summary_df = pd.DataFrame(global_summary)
summary_df.to_csv(
    os.path.join(OUTPUT_DIR, "RF_summary_confidence_intervals.csv"),
    index=False
)

print("Random Forest analysis with confidence intervals completed successfully.")
