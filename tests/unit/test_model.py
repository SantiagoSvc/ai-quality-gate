import pytest
from src.model import predict, train_model

# ================================================
# FIXTURE: código que se ejecuta antes de las pruebas
# El modelo se entrena una sola vez y se comparte
# ================================================
@pytest.fixture(scope="module")
def model():
    return train_model()


# ================================================
# PRUEBAS DE COMPORTAMIENTO BÁSICO
# ¿El modelo responde lo que debería?
# ================================================
def test_positive_sentiment(model):
    result = predict("This product is amazing", model)
    assert result["sentiment"] == "positive"

def test_negative_sentiment(model):
    result = predict("Terrible experience", model)
    assert result["sentiment"] == "negative"

def test_result_has_required_fields(model):
    result = predict("Great service", model)
    # Verificamos que el resultado tenga las 3 claves esperadas
    assert "text" in result
    assert "sentiment" in result
    assert "confidence" in result

def test_confidence_is_valid_range(model):
    result = predict("I love this", model)
    # La confianza debe ser un número entre 0 y 1
    assert 0.0 <= result["confidence"] <= 1.0

def test_sentiment_is_valid_label(model):
    result = predict("Good product", model)
    # Solo pueden existir estos dos valores
    assert result["sentiment"] in ["positive", "negative"]


# ================================================
# PRUEBAS DE EDGE CASES
# ¿Qué pasa con inputs raros o extremos?
# Esto es pensar como QA
# ================================================
def test_empty_string(model):
    # Un string vacío no debe romper el sistema
    result = predict("", model)
    assert result["sentiment"] in ["positive", "negative"]

def test_single_word(model):
    result = predict("good", model)
    assert result["sentiment"] in ["positive", "negative"]

def test_very_long_text(model):
    long_text = "good " * 500
    result = predict(long_text, model)
    assert result["sentiment"] in ["positive", "negative"]

def test_numbers_as_text(model):
    # ¿Qué pasa si alguien manda solo números?
    result = predict("12345", model)
    assert result["sentiment"] in ["positive", "negative"]

def test_special_characters(model):
    result = predict("!!! @@@ ###", model)
    assert result["sentiment"] in ["positive", "negative"]

def test_text_is_returned_unchanged(model):
    text = "Amazing product"
    result = predict(text, model)
    # El campo text debe devolver exactamente lo que se envió
    assert result["text"] == text