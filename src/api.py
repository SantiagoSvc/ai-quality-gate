from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.model import predict, load_model

# ================================================
# Inicializar la aplicación FastAPI
# ================================================
app = FastAPI(
    title="AI Quality Gate API",
    description="API para clasificación de sentimientos con testing automatizado",
    version="1.0.0"
)

# ================================================
# Cargar el modelo al iniciar la API
# Así no lo carga en cada petición, sino una sola vez
# ================================================
model = None

@app.on_event("startup")
def startup_event():
    global model
    model = load_model()
    print("✅ Modelo cargado correctamente")


# ================================================
# Esquema de entrada
# Pydantic valida que el request tenga el formato correcto
# ================================================
class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float


# ================================================
# ENDPOINTS
# ================================================

@app.get("/health")
def health_check():
    """Verifica que la API está viva"""
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictResponse)
def predict_sentiment(request: PredictRequest):
    """Recibe un texto y devuelve el sentimiento"""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="El texto no puede estar vacío")
    
    result = predict(request.text, model)
    return result


@app.get("/")
def root():
    return {"message": "AI Quality Gate API", "docs": "/docs"}