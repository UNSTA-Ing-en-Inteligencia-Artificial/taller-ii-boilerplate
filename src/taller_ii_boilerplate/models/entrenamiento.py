import argparse
import os
from collections import Counter

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.utils import resample


def cargar_dataset(dataset_path, nombre_predictor, etiquetas_path):
    X = pd.read_csv(dataset_path)
    if nombre_predictor in X.columns:
        y = X.pop(nombre_predictor)
        return X, y

    etiquetas = pd.read_csv(etiquetas_path)
    y = etiquetas[nombre_predictor].reset_index(drop=True)
    return X.reset_index(drop=True), y


def balancear_clases(df, nombre_predictor, random_state=42):
    conteos = df[nombre_predictor].value_counts()
    objetivo = int(conteos.median())
    partes = []
    for clase, cantidad in conteos.items():
        subset = df[df[nombre_predictor] == clase]
        if cantidad > objetivo:
            subset = resample(subset, replace=False, n_samples=objetivo, random_state=random_state)
        elif cantidad < objetivo:
            subset = resample(subset, replace=True, n_samples=objetivo, random_state=random_state)
        partes.append(subset)
    return pd.concat(partes)


def entrenar_y_evaluar(modelo, X_train, y_train, X_test, y_test, nombre):
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    print(f"\n=== {nombre} ===")
    print(classification_report(y_test, y_pred, digits=3, zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=modelo.classes_)
    ConfusionMatrixDisplay(cm, display_labels=modelo.classes_).plot(cmap="Blues")
    plt.title(f"Matriz de confusión - {nombre}")
    plt.show()
    plt.close()

    return f1_score(y_test, y_pred, average="macro")


dataset = "data/final/dataset_bow_unigram.csv"
etiquetas_dataset = "data/preprocessed/dataset.csv"
columna_y = "Rating"
balancear = True
tamanio_test = 0.2

X, y = cargar_dataset(dataset, columna_y, etiquetas_dataset)
print("Tamaño del dataset vectorizado:", X.shape)
print("Distribución de clases:\n", y.value_counts())

df = X.copy()
df[columna_y] = y.values
if balancear:
    df = balancear_clases(df, columna_y)

X_bal = df.drop(columns=[columna_y])
y_bal = df[columna_y]

print("Tamaño del dataset vectorizado (balanceado):", X_bal.shape)
print("Distribución de clases (balanceado):\n", y_bal.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X_bal,
    y_bal,
    test_size=tamanio_test,
    random_state=42,
    stratify=y_bal,
)

# Baseline
majority_class = Counter(y_train).most_common(1)[0][0]
y_pred_baseline = [majority_class] * len(y_test)
print("\n=== Baseline (clase mayoritaria) ===")
print(classification_report(y_test, y_pred_baseline, zero_division=0))

os.makedirs("data/models", exist_ok=True)
nombre_base = os.path.splitext(os.path.basename(dataset))[0]

modelos = {
    "Regresión Logística": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "SVM Lineal": LinearSVC(class_weight="balanced"),
}
# MultinomialNB no admite features negativas (puede pasar con un dataset vectorizado con embeddings)
if (X_bal.values >= 0).all():
    modelos["Naive Bayes"] = MultinomialNB()

resultados = {}
for nombre, modelo in modelos.items():
    slug = nombre.lower().replace(" ", "_")
    f1 = entrenar_y_evaluar(
        modelo, X_train, y_train, X_test, y_test, nombre
    )
    resultados[nombre] = f1

    ruta_modelo = os.path.join("data/models", f"{nombre_base}_{slug}.pkl")
    joblib.dump(modelo, ruta_modelo)
    print(f"Modelo guardado en {ruta_modelo}")

tabla = pd.DataFrame(
    {"Modelo": list(resultados.keys()), "Macro-F1": list(resultados.values())}
)
print("\n=== Comparación de modelos ===")
print(tabla.sort_values(by="Macro-F1", ascending=False))
