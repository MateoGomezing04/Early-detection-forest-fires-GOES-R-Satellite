
#Codigo para modelo Random Forest


import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, f1_score, precision_score, recall_score
)
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
FILE_NAME = r"C:\Users\User\Desktop\Articulo.xlsx"
SHEET_NAME = "Sheet1"
VALIDATION_SIZE = 0.3
RANDOM_SEED = 42
MIN_SAMPLES = 30
OUTPUT_DIR = r"C:\Users\User\Desktop\Articulo\RF_Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- CARGA ----------
df = pd.read_excel(FILE_NAME, sheet_name=SHEET_NAME)
df.columns = df.columns.str.strip()

# Detectar columnas clave
product_col_candidates = ['PRODUCTO', 'Producto', 'producto', 'PRODUCT', 'PRODUCT_NAME']
product_col = next((c for c in product_col_candidates if c in df.columns), None)
if product_col is None:
    raise ValueError("No se encontró columna de producto ('PRODUCTO' o similar).")

valor_col_candidates = ['VALOR', 'Valor', 'valor', 'Value']
valor_col = next((c for c in valor_col_candidates if c in df.columns), None)
if valor_col is None:
    raise ValueError("No se encontró columna 'VALOR' en el Excel.")

y_candidates = ['y_grid', 'y', 'Y', 'lat', 'LAT']
x_candidates = ['x_grid', 'x', 'X', 'lon', 'LON']
y_col = next((c for c in y_candidates if c in df.columns), None)
x_col = next((c for c in x_candidates if c in df.columns), None)

target_candidates = ['INCENDIO', 'Incendio', 'incendio']
target_col = next((c for c in target_candidates if c in df.columns), None)
if target_col is None:
    raise ValueError("No se encontró la columna objetivo 'INCENDIO' en el Excel.")

# ---------- ITERAR POR PRODUCTO ----------
productos = df[product_col].unique()
print(f"Productos detectados ({len(productos)}): {productos}")
resumenes_globales = []

for prod in productos:
    subset = df[df[product_col] == prod].copy()
    n = len(subset)
    print(f"\n--- Procesando producto: {prod} (n = {n}) ---")
    if n < MIN_SAMPLES:
        print(f"  Saltado: menos de {MIN_SAMPLES} muestras.")
        continue

    if subset[target_col].nunique() < 2:
        print("  Saltado: no contiene ambas clases (0 y 1).")
        continue

    predictors = [valor_col]
    if y_col is not None and x_col is not None:
        predictors += [y_col, x_col]
    predictors = [p for p in predictors if p in subset.columns]

    X = subset[predictors].copy()
    y = subset[target_col].fillna(0).astype(int)
    X = X.fillna(X.mean())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y,
            test_size=VALIDATION_SIZE,
            random_state=RANDOM_SEED,
            stratify=y
        )
    except Exception as e:
        print(f"  Error en train_test_split: {e}")
        continue

    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, class_weight='balanced')
    rf.fit(X_train, y_train)
    y_pred_test = rf.predict(X_test)
    y_proba_test = rf.predict_proba(X_test)[:, 1] if hasattr(rf, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred_test)
    prec = precision_score(y_test, y_pred_test, zero_division=0)
    rec = recall_score(y_test, y_pred_test, zero_division=0)
    f1 = f1_score(y_test, y_pred_test, zero_division=0)
    auc = roc_auc_score(y_test, y_proba_test) if y_proba_test is not None else np.nan

    print(f"  Accuracy: {acc:.4f}  Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}  AUC: {auc if not np.isnan(auc) else 'N/A'}")

    prod_safe = str(prod).replace(" ", "_").replace("/", "_")
    out_dir = os.path.join(OUTPUT_DIR, prod_safe)
    os.makedirs(out_dir, exist_ok=True)

    # ---- Matriz de confusión ----
    cm = confusion_matrix(y_test, y_pred_test)
    plt.figure(figsize=(4,4))
    plt.imshow(cm, interpolation='nearest')
    plt.title(f"Confusion matrix - {prod}")
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('Real')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i,j] > cm.max()/2 else "black")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150)
    plt.close()

    # ---- ROC y AUC ----
    if y_proba_test is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba_test)
        plt.figure(figsize=(4,4))
        plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
        plt.plot([0,1],[0,1],'k--')
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title(f"ROC - {prod}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "roc_curve.png"), dpi=150)
        plt.close()

    # ---- Precision-Recall ----
    if y_proba_test is not None:
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba_test)
        plt.figure(figsize=(4,4))
        plt.plot(recall_vals, precision_vals)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall - {prod}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "precision_recall.png"), dpi=150)
        plt.close()

    # ---- Importancia de variables ----
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
    plt.figure(figsize=(5, max(2, len(importances)*0.4)))
    importances.plot(kind='barh')
    plt.title(f"Importance - {prod}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "feature_importances.png"), dpi=150)
    plt.close()

    # ---- Guardar resumen individual ----
    resumen_df = pd.DataFrame({
        "product": [prod],
        "n_samples": [n],
        "accuracy": [acc],
        "precision": [prec],
        "recall": [rec],
        "f1": [f1],
        "auc": [auc]
    })
    resumen_df.to_csv(os.path.join(out_dir, "resumen_metricas.csv"), index=False)

    resumenes_globales.append(resumen_df)

# ---------- GUARDAR RESUMEN GLOBAL ----------
if resumenes_globales:
    resumen_global = pd.concat(resumenes_globales, ignore_index=True)
    resumen_global.to_csv(os.path.join(OUTPUT_DIR, "resumen_global.csv"), index=False)
    print(f"\n Resumen global guardado en: {os.path.join(OUTPUT_DIR, 'resumen_global.csv')}")
    
    # ---------- IDENTIFICAR EL MEJOR PRODUCTO ----------
try:
    # Criterio principal: F1-Score (puedes cambiarlo a 'auc' si prefieres)
    mejor_producto = resumen_global.loc[resumen_global["f1"].idxmax()]
    print("\n El mejor producto para predecir incendios es:")
    print(mejor_producto[["producto", "f1", "accuracy", "precision", "recall", "auc"]])

    # Guardar también en un archivo de texto
    with open(os.path.join(OUTPUT_DIR, "mejor_producto.txt"), "w") as f:
        f.write("=== MEJOR PRODUCTO PARA PREDICCIÓN DE INCENDIOS ===\n\n")
        f.write(f"Producto: {mejor_producto['producto']}\n")
        f.write(f"F1-Score: {mejor_producto['f1']:.4f}\n")
        f.write(f"Accuracy: {mejor_producto['accuracy']:.4f}\n")
        f.write(f"Precision: {mejor_producto['precision']:.4f}\n")
        f.write(f"Recall: {mejor_producto['recall']:.4f}\n")
        f.write(f"AUC: {mejor_producto['auc']:.4f}\n")
    print(f"\n Archivo con el mejor producto guardado en: {os.path.join(OUTPUT_DIR, 'mejor_producto.txt')}")
except Exception as e:
    print(f"\n No se pudo determinar el mejor producto: {e}")
    # ---------- GRÁFICA COMPARATIVA DE PRODUCTOS ----------
    try:
        plt.figure(figsize=(10, 6))
        resumen_plot = resumen_global.sort_values(by="f1", ascending=False)
        plt.bar(resumen_plot["producto"], resumen_plot["f1"], color="steelblue", label="F1-Score")
        plt.xlabel("Producto satelital")
        plt.ylabel("F1-Score")
        plt.title("Comparación de desempeño por producto")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.legend()
        plt.savefig(os.path.join(OUTPUT_DIR, "comparacion_productos.png"), dpi=200)
        plt.close()

        print(f" Gráfica comparativa guardada en: {os.path.join(OUTPUT_DIR, 'comparacion_productos.png')}")
    except Exception as e:
        print(f" No se pudo generar la gráfica comparativa: {e}")

else:
    print("\n No se generó ningún resumen global (posiblemente no hubo productos válidos).")
