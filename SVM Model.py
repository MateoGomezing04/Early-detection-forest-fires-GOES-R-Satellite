# ==========================================================
# SVM WITH 95% CONFIDENCE INTERVALS (PERCENTILES) BY PRODUCT
# ==========================================================
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, precision_recall_curve
)

warnings.filterwarnings("ignore")

# ==========================================================
# CONFIGURATION
# ==========================================================

DATA_FILE = r"C:\Users\User\Desktop\Articulo\Consolidado manual datos dia antes y dia del incendio 12h.xlsx"
OUTPUT_DIR = os.path.join(os.path.dirname(DATA_FILE), "SVM_Results_CI_recuperado_301")
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_ITERATIONS = 50
TEST_SIZE = 0.30
RANDOM_SEED_BASE = 100
MIN_SAMPLES = 30

# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_excel(DATA_FILE)
df.columns = df.columns.str.replace('[^A-Za-z0-9_]+', '', regex=True)

required_cols = {'PRODUCTO', 'INCENDIO', 'Valor', 'x', 'y'}
if not required_cols.issubset(df.columns):
    raise ValueError(f"Missing columns: {required_cols - set(df.columns)}")

products = df['PRODUCTO'].unique()
global_summary = []

print(f"\n📊 {len(products)} products detected\n")

# ==========================================================
# LOOP BY PRODUCT
# ==========================================================

for product in tqdm(products, desc="Processing products", unit="product"):


    subset = df[df['PRODUCTO'] == product].copy()
    n_samples = len(subset)

    if n_samples < MIN_SAMPLES or subset['INCENDIO'].nunique() < 2:
        if n_samples < MIN_SAMPLES:
            print(f"⛔ {product}: descartado por pocas muestras ({n_samples})")
            continue

        if subset['INCENDIO'].nunique() < 2:
            print(f"⛔ {product}: descartado por una sola clase INCENDIO")
            continue

        continue

    X = subset[['Valor', 'x', 'y']].fillna(subset[['Valor', 'x', 'y']].mean())
    y = subset['INCENDIO'].astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    metrics_iterations = []

    # ======================================================
    # ITERATIONS
    # ======================================================

    for i in range(N_ITERATIONS):

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled,
            y,
            test_size=TEST_SIZE,
            stratify=y,
            random_state=RANDOM_SEED_BASE + i
        )

        svm = SVC(
            kernel='linear',
            C=1.0,
            probability=True,
            class_weight='balanced',
            random_state=RANDOM_SEED_BASE + i
        )

        svm.fit(X_train, y_train)

        y_pred = svm.predict(X_test)
        y_prob = svm.predict_proba(X_test)[:, 1]

        metrics_iterations.append({
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "auc": roc_auc_score(y_test, y_prob)
        })

        # Guardar última iteración para curvas clásicas
        if i == N_ITERATIONS - 1:
            X_train_last = X_train
            X_test_last = X_test
            y_train_last = y_train
            y_test_last = y_test
            y_pred_last = y_pred
            y_prob_last = y_prob

    metrics_df = pd.DataFrame(metrics_iterations)

    # ======================================================
    # CONFIDENCE INTERVALS (95%)
    # ======================================================

    summary = {
        "product": product,
        "n_samples": n_samples
    }

    for metric in metrics_df.columns:
        summary[f"{metric}_mean"] = metrics_df[metric].mean()
        summary[f"{metric}_p2_5"] = np.percentile(metrics_df[metric], 2.5)
        summary[f"{metric}_p97_5"] = np.percentile(metrics_df[metric], 97.5)

    global_summary.append(summary)

    # ======================================================
    # OUTPUT DIRECTORY
    # ======================================================

    prod_safe = str(product).replace(" ", "_").replace("/", "_")
    prod_dir = os.path.join(OUTPUT_DIR, prod_safe)
    os.makedirs(prod_dir, exist_ok=True)

    metrics_df.to_csv(
        os.path.join(prod_dir, "metrics_iterations.csv"),
        index=False
    )

    # ======================================================
    # ERROR CURVE VS C
    # ======================================================

    C_values = np.logspace(-3, 3, 10)
    train_errors = []
    test_errors = []

    for C in C_values:

        svm_tmp = SVC(
            kernel='linear',
            C=C,
            class_weight='balanced',
            random_state=RANDOM_SEED_BASE
        )

        svm_tmp.fit(X_train_last, y_train_last)

        train_errors.append(
            1 - accuracy_score(y_train_last, svm_tmp.predict(X_train_last))
        )
        test_errors.append(
            1 - accuracy_score(y_test_last, svm_tmp.predict(X_test_last))
        )

    plt.figure(figsize=(6, 4))
    plt.semilogx(C_values, train_errors, marker='o', label='Training error')
    plt.semilogx(C_values, test_errors, marker='s', label='Test error')
    plt.xlabel('Regularization parameter C')
    plt.ylabel('Classification error')
    plt.title(f'SVM Error vs C – {product}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(prod_dir, "svm_error_curve_C.png"), dpi=200)
    plt.close()

    # ======================================================
    # CONFUSION MATRIX
    # ======================================================

    cm = confusion_matrix(y_test_last, y_pred_last)
    plt.figure(figsize=(4, 4))
    plt.imshow(cm)
    plt.title(f"Confusion Matrix – {product}")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Observed")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    plt.savefig(os.path.join(prod_dir, "confusion_matrix.png"), dpi=200)
    plt.close()

    # ======================================================
    # ROC CURVE
    # ======================================================

    fpr, tpr, _ = roc_curve(y_test_last, y_prob_last)
    plt.figure(figsize=(4, 4))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc_score(y_test_last, y_prob_last):.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC – {product}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(prod_dir, "roc_curve.png"), dpi=200)
    plt.close()

    # ======================================================
    # PRECISION–RECALL CURVE
    # ======================================================

    prec, rec, _ = precision_recall_curve(y_test_last, y_prob_last)
    plt.figure(figsize=(4, 4))
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision–Recall – {product}")
    plt.tight_layout()
    plt.savefig(os.path.join(prod_dir, "precision_recall.png"), dpi=200)
    plt.close()

    print(f"✅ {product}: completed")

# ==========================================================
# GLOBAL SUMMARY
# ==========================================================

summary_df = pd.DataFrame(global_summary)
summary_df.to_csv(
    os.path.join(OUTPUT_DIR, "SVM_summary_confidence_intervals.csv"),
    index=False
)

print("\n📈 SVM analysis with confidence intervals completed successfully.")
