from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pickle
import os

# ¿Qué es un Pipeline?
# Es una cadena que primero convierte texto a números (TF-IDF)
# y luego aplica el modelo de clasificación.
# Así tratamos ambos pasos como uno solo.

def train_model():
    # Datos de entrenamiento: frases y su etiqueta (1=positivo, 0=negativo)
    texts = [
        "I love this product",
        "This is amazing",
        "Excellent quality",
        "Best experience ever",
        "Highly recommended",
        "Great service",
        "I hate this",
        "Terrible quality",
        "Worst experience ever",
        "Do not recommend",
        "Very disappointed",
        "Awful product",
    ]

    labels = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]

    # Crear el pipeline
    model = Pipeline([
        ("vectorizer", TfidfVectorizer()),
        ("classifier", LogisticRegression())
    ])

    # Entrenar
    model.fit(texts, labels)

    # Guardar el modelo entrenado en un archivo .pkl
    os.makedirs("models", exist_ok=True)
    with open("models/sentiment_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("✅ Modelo entrenado y guardado en models/sentiment_model.pkl")
    return model


def load_model():
    # Carga el modelo desde el archivo .pkl
    with open("models/sentiment_model.pkl", "rb") as f:
        return pickle.load(f)


def predict(text: str, model=None) -> dict:
    # Si no se pasa un modelo, carga el guardado
    if model is None:
        model = load_model()

    prediction = model.predict([text])[0]
    probability = model.predict_proba([text])[0]

    return {
        "text": text,
        "sentiment": "positive" if prediction == 1 else "negative",
        "confidence": round(float(max(probability)), 4)
    }


# Esto permite correr el archivo directamente para entrenar
if __name__ == "__main__":
    train_model()