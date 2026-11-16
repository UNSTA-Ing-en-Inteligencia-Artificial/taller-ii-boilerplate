import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import joblib
from pathlib import Path

if __name__ == "__main__":
    df = pd.read_csv("data/preprocessed/dataset.csv")

    # Separamos un conjunto de test que no se toca hasta el final
    X_train, X_test, y_train, y_test = train_test_split(
        df["review_clean"], df["Rating"],
        test_size=0.2, random_state=42, stratify=df["Rating"],
    )

    def optimizar_modelo(modelo, hiperparametros, X_train, y_train):
        # El vectorizador va dentro del Pipeline y se reajusta en cada fold (para evitar peeking)
        pipe = Pipeline([
            ("bow", CountVectorizer(max_features=10000, ngram_range=(1, 2))),
            ("clf", modelo),
        ])
        grid_search = GridSearchCV(
            pipe, hiperparametros,
            cv=5, scoring="f1_macro", verbose=2, n_jobs=-1,
        )
        grid_search.fit(X_train, y_train)
        print("Mejores hiperparámetros:", grid_search.best_params_)
        print("Mejor F1 macro (CV):", grid_search.best_score_)
        return grid_search

    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    grid_lr = optimizar_modelo(lr, {
        "clf__C": [0.01, 0.1, 1, 10],
        "clf__solver": ["lbfgs"],
    }, X_train, y_train)

    svm = LinearSVC(class_weight="balanced", max_iter=5000, random_state=42)
    grid_svm = optimizar_modelo(svm, [
        {"clf__penalty": ["l2"], "clf__loss": ["hinge", "squared_hinge"], "clf__C": np.linspace(0.01, 10)},
        {"clf__penalty": ["l1"], "clf__loss": ["squared_hinge"], "clf__dual": [False], "clf__C": [0.01, 0.1, 1, 10]},
    ], X_train, y_train)

    # Evaluación final en test, una sola vez
    print(classification_report(y_test, grid_lr.predict(X_test)))
    print(classification_report(y_test, grid_svm.predict(X_test)))

    # Guardamos vectorizador + modelo para inferencia
    output_dir = Path("data/models")
    # Logistic Regression
    best_lr = grid_lr.best_estimator_
    joblib.dump(best_lr.named_steps["bow"],output_dir / "lr_vectorizer.pkl")
    joblib.dump(best_lr.named_steps["clf"],output_dir / "lr_classifier.pkl")

    # SVM
    best_svm = grid_svm.best_estimator_
    joblib.dump(best_svm.named_steps["bow"], output_dir / "svm_vectorizer.pkl")
    joblib.dump(best_svm.named_steps["clf"], output_dir / "svm_classifier.pkl" )