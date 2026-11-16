import joblib
import gradio as gr
from taller_ii_boilerplate.inference.inferencia import predecir_sentimiento

vectorizador = joblib.load("data/models/lr_vectorizer.pkl")
modelo = joblib.load("data/models/lr_classifier.pkl")

def interfaz(texto):
    resultado = predecir_sentimiento(texto, vectorizador, modelo)
    return resultado.get("prediccion"), resultado.get("probabilidades", "{}")



demo = gr.Interface(
    theme=gr.Theme.from_hub("allenai/gradio-theme"),
    fn=interfaz,
    inputs=gr.Textbox(label="Opinión", lines=3, placeholder="Escribe una texto aquí..."),
    outputs=[
        gr.Label(label="Sentimiento predicho"),
        gr.JSON(label="Probabilidades del modelo")
    ],
    title="Análisis de sentimiento",
    description="Escribe un texto y el modelo clasificará su sentimiento como positivo, negativo o neutro."
)

demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
