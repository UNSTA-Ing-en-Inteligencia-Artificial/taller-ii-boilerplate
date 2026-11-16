import joblib
from taller_ii_boilerplate.preprocessing.procesamiento import preprocesar_texto
import numpy as np

def predecir_sentimiento(opinion, vectorizador, modelo):
    opinion_preprocesada = preprocesar_texto(opinion)
    vector = vectorizador.transform([opinion_preprocesada])
    prediccion = modelo.predict(vector)[0]
    probabilidades = modelo.predict_proba(vector)[0]
    probabilidades_limpias = {str(k): v for k, v in dict(zip(modelo.classes_, np.round(probabilidades, 3))).items()}
    return {"texto": opinion, "prediccion": prediccion, "probabilidades": probabilidades_limpias}

if __name__ == "__main__":
    vectorizador = joblib.load("data/models/lr_vectorizer.pkl")
    modelo = joblib.load("data/models/lr_classifier.pkl")
    ejemplo = "excellent customer service, they fixed my problem quickly"
    resultado = predecir_sentimiento(ejemplo, vectorizador, modelo)
    print(resultado)
