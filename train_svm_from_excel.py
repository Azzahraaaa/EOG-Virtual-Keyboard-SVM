# train_svm_from_excel.py
# Latih SVM dari Excel (DataTraining.xlsx) dan simpan hasil + grafik

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_curve, auc
)

# ===================== KONFIGURASI =====================
EXCEL_PATH   = "DataTraining.xlsx"
LABEL_COL    = "label"
TEST_SIZE    = 0.25
RANDOM_STATE = 42
MODEL_OUT    = "model_svm.pkl"
SVM_C        = 3.0
SVM_GAMMA    = "scale"
# =======================================================

def normalize_labels(y_raw):
    y = []
    for v in y_raw:
        if isinstance(v, str):
            s = v.strip().lower()
            if "atas" in s:
                y.append(1)
            elif "kedip" in s:
                y.append(2)
            else:
                raise ValueError(f"Label tidak dikenali: {v}")
        else:
            y.append(int(v))
    return np.asarray(y, dtype=int)

def main():
    if not os.path.exists(EXCEL_PATH):
        raise FileNotFoundError(f"File tidak ditemukan: {EXCEL_PATH}")

    df = pd.read_excel(EXCEL_PATH)
    df = df.dropna(subset=["ch1", "ch2", LABEL_COL])
    X = df[["ch1", "ch2"]].to_numpy(dtype=float)
    y = normalize_labels(df[LABEL_COL].to_numpy())

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", C=SVM_C, gamma=SVM_GAMMA, probability=True, random_state=RANDOM_STATE)),
    ])
    pipeline.fit(X_tr, y_tr)

    y_pred = pipeline.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    print(f"Akurasi validasi: {acc:.4f}")
    print("Confusion matrix:\n", confusion_matrix(y_te, y_pred))
    print("\nClassification report:\n", classification_report(y_te, y_pred))

    # ---------- Simpan confusion_matrix.png ----------
    cm = confusion_matrix(y_te, y_pred)
    plt.imshow(cm, cmap="viridis")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.savefig("confusion_matrix.png")
    plt.close()

    # ---------- Simpan decision_region.png ----------
    X_combined = np.vstack((X_tr, X_te))
    y_combined = np.hstack((y_tr, y_te))
    x_min, x_max = X_combined[:, 0].min() - 1, X_combined[:, 0].max() + 1
    y_min, y_max = X_combined[:, 1].min() - 1, X_combined[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    Z = pipeline.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X_combined[:, 0], X_combined[:, 1], c=y_combined, edgecolors="k")
    plt.title("Decision Region")
    plt.savefig("decision_region.png")
    plt.close()

    # ---------- Simpan ROC Curve ----------
    y_score = pipeline.decision_function(X_te)
    fpr, tpr, _ = roc_curve(y_te, y_score)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0,1],[0,1],"--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig("roc_curve.png")
    plt.close()

    # ---------- Scatter latih & uji ----------
    plt.scatter(X_tr[:, 0], X_tr[:, 1], c=y_tr, marker="o", label="Train")
    plt.scatter(X_te[:, 0], X_te[:, 1], c=y_te, marker="x", label="Test")
    plt.xlabel("CH1")
    plt.ylabel("CH2")
    plt.legend()
    plt.title("Scatter Data Latih & Uji")
    plt.savefig("scatter_latih_uji.png")
    plt.close()

    # ---------- Simpan prediksi ke Excel ----------
    df_pred = pd.DataFrame({
        "ch1": X_te[:, 0],
        "ch2": X_te[:, 1],
        "label_asli": y_te,
        "label_pred": y_pred
    })
    df_pred.to_excel("prediksi_uji.xlsx", index=False)

    # ---------- Simpan model ----------
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(pipeline, f)

    print(f"\nModel tersimpan ke: {os.path.abspath(MODEL_OUT)}")

if __name__ == "__main__":
    main()
