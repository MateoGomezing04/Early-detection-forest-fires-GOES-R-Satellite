# ==========================================================
# SVM MODEL (Support Vector Machine) BY PRODUCT
# ==========================================================

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, zero_one_loss
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# ==========================================================
# 1 INITIAL CONFIGURATION
# ==========================================================
data_file = r"C:\Users\User\Desktop\Articuloxlsx"
out_dir = os.path.join(os.path.dirname(data_file), "SVM_Results")
os.makedirs(out_dir, exist_ok=True)

VALIDATION_SIZE = 0.3
RANDOM_SEED = 42

# ==========================================================
# 2 DATA LOADING
# ==========================================================
df = pd.read_excel(data_file)
df.columns = df.columns.str.replace('[^A-Za-z0-9_]+', '', regex=True)

required_cols = {'PRODUCTO', 'INCENDIO', 'Valor', 'x', 'y'}
if not required_cols.issubset(df.columns):
    raise ValueError(f"Missing required columns: {required_cols - set(df.columns)}")

# ==========================================================
# 3 PROCESSING BY PRODUCT
# ==========================================================
products = df['PRODUCTO'].unique()
summary_metrics = []

print(f"\n📊 {len(products)} products were found in the dataset.\n")

for product in products:

    subset = df[df['PRODUCTO'] == product].copy()

    if subset['INCENDIO'].nunique() < 2:
        print(f"Product {product}: not enough classes available.\n")
        continue

    X = subset[['Valor', 'x', 'y']].fillna(subset[['Valor', 'x', 'y']].mean())
    y = subset['INCENDIO'].fillna(0).astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_SEED,
        stratify=y
    )

    # ==========================================================
    # 4 MAIN MODEL TRAINING
    # ==========================================================
    model = SVC(
        kernel='linear',
        probability=True,
        class_weight='balanced',
        random_state=RANDOM_SEED
    )
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_prob_test = model.predict_proba(X_test)[:, 1]

    # ==========================================================
    # 5 PERFORMANCE METRICS
    # ==========================================================
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)

    report = classification_report(
        y_test, y_pred_test,
        output_dict=True, zero_division=0
    )

    precision = report['1']['precision']
    recall = report['1']['recall']
    f1 = report['1']['f1-score']
    auc = roc_auc_score(y_test, y_prob_test)
    cm = confusion_matrix(y_test, y_pred_test)
    # ==========================================================
    # 6.1 INDIVIDUAL METRICS PLOT PER PRODUCT
    # ==========================================================
    metrics_names = [
        'Train Accuracy',
        'Test Accuracy',
        'Precision',
        'Recall',
        'F1-score',
        'AUC'
    ]
    
    metrics_values = [
        acc_train,
        acc_test,
        precision,
        recall,
        f1,
        auc
    ]
    
    plt.figure(figsize=(7, 4))
    sns.barplot(
        x=metrics_names,
        y=metrics_values
    )
    plt.ylim(0, 1)
    plt.ylabel('Metric value')
    plt.title(f'SVM Performance Metrics - {product}')
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, f"Metrics_{product}.png"),
        dpi=300
    )
    plt.close()

    # ==========================================================
    # 6 ERROR CURVE VS C PARAMETER (OPTION 1)
    # ==========================================================
    C_values = np.logspace(-3, 3, 10)
    train_errors = []
    test_errors = []

    for C in C_values:
        svm_tmp = SVC(
            kernel='linear',
            C=C,
            class_weight='balanced',
            random_state=RANDOM_SEED
        )
        svm_tmp.fit(X_train, y_train)

        train_errors.append(
            zero_one_loss(y_train, svm_tmp.predict(X_train))
        )
        test_errors.append(
            zero_one_loss(y_test, svm_tmp.predict(X_test))
        )

    plt.figure(figsize=(6, 4))
    plt.semilogx(C_values, train_errors, marker='o', label='Training error')
    plt.semilogx(C_values, test_errors, marker='s', label='Validation error')
    plt.xlabel('Regularization parameter C')
    plt.ylabel('Classification error')
    plt.title(f'SVM Error Curve vs C - {product}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"Error_Curve_C_{product}.png"), dpi=300)
    plt.close()

    # ==========================================================
    # 7 SAVE RESULTS
    # ==========================================================
    details_path = os.path.join(out_dir, f"Results_{product}.xlsx")

    metrics_df = pd.DataFrame({
        'Metric': [
            'Training Accuracy', 'Validation Accuracy',
            'Precision (Fire)', 'Recall (Fire)',
            'F1-score (Fire)', 'AUC'
        ],
        'Value': [acc_train, acc_test, precision, recall, f1, auc]
    })

    cm_df = pd.DataFrame(
        cm,
        index=['Actual_NoFire', 'Actual_Fire'],
        columns=['Predicted_NoFire', 'Predicted_Fire']
    )

    with pd.ExcelWriter(details_path, engine='openpyxl') as writer:
        metrics_df.to_excel(writer, sheet_name='Summary', index=False)
        pd.DataFrame(report).transpose().to_excel(writer, sheet_name='Classification_Report')
        cm_df.to_excel(writer, sheet_name='Confusion_Matrix')

    summary_metrics.append({
        'Product': product,
        'Accuracy_Test': acc_test,
        'F1_Fire': f1,
        'AUC': auc
    })

    print(f"✅ {product}: processed successfully.\n")

# ==========================================================
# 8 GLOBAL SUMMARY
# ==========================================================
if summary_metrics:
    summary_df = pd.DataFrame(summary_metrics)
    summary_df.to_excel(
        os.path.join(out_dir, "Global_SVM_Summary.xlsx"),
        index=False
    )

    plt.figure(figsize=(10, 6))
    summary_df.plot(
        x='Product',
        y=['Accuracy_Test', 'F1_Fire', 'AUC'],
        kind='bar'
    )
    plt.ylabel('Metric value')
    plt.ylim(0, 1)
    plt.title('Global Comparison of SVM Performance Metrics')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "Global_SVM_Metrics.png"), dpi=300)
    plt.close()

    print("📈 Global summary generated successfully.\n")
