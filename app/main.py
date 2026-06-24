import sys
from pathlib import Path

# Add project root and app folder to Python search path dynamically
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from contextlib import asynccontextmanager
from fastapi import FastAPI
from ml.inference import predictor

try:
    from app.api.endpoints import router
except ImportError:
    from api.endpoints import router



@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: modeli önceden yükle
    predictor.load()
    yield
    # Shutdown: gerekirse temizlik yapılabilir


app = FastAPI(
    title="Credit Risk Analytics API",
    description="XGBoost tabanlı kredi riski tahmin API'si. FastAPI + Scikit-learn Pipeline entegrasyonu.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)


@app.get("/", tags=["health"])
def read_root():
    return {"message": "Credit Risk Analytics API is running", "version": "1.0.0"}


@app.get("/health", tags=["health"])
def health_check():
    return {"status": "ok", "model_loaded": predictor.is_loaded}