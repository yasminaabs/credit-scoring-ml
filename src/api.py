"""
API FastAPI pour Credit Scoring (Partie I - Q9)

Lancement depuis la racine du projet :
    python -m uvicorn api:app --app-dir src --reload

Documentation interactive : http://127.0.0.1:8000/docs
"""

import os
import pickle
from pathlib import Path
from typing import List

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Charger le pipeline généré par la Q8 (models/final_pipeline_auto.pkl).
# Un autre modèle peut être utilisé via la variable d'environnement CREDIT_MODEL_PATH.
DEFAULT_PIPELINE_PATH = Path(__file__).resolve().parent.parent / 'models' / 'final_pipeline_auto.pkl'
PIPELINE_PATH = Path(os.environ.get('CREDIT_MODEL_PATH', DEFAULT_PIPELINE_PATH))

if not PIPELINE_PATH.exists():
    raise FileNotFoundError(
        f"Pipeline introuvable: {PIPELINE_PATH} "
        "(exécutez le notebook jusqu'à la Q8 pour le générer)"
    )

with open(PIPELINE_PATH, 'rb') as f:
    pipeline = pickle.load(f)

# Ordre des colonnes de credit_scoring.csv (sans Status), attendu par le pipeline
FEATURES = ['seniority', 'home', 'time', 'age', 'marital', 'records', 'job',
            'expenses', 'income', 'assets', 'debt', 'amount', 'price']


# Modèle de données
class CreditRequest(BaseModel):
    seniority: float
    home: float
    time: float
    age: float
    marital: float
    records: float
    job: float
    expenses: float
    income: float
    assets: float
    debt: float
    amount: float
    price: float


# Application FastAPI
app = FastAPI(
    title="API Credit Scoring",
    description="Prédiction d'acceptation de crédit",
    version="1.0.0"
)


def predire(clients: List[CreditRequest]) -> List[dict]:
    """Prédictions pour une liste de clients (un seul appel au pipeline)"""
    X = np.array([[getattr(client, feature) for feature in FEATURES] for client in clients])

    preds = pipeline.predict(X)
    probas = pipeline.predict_proba(X)

    # Colonnes de predict_proba dans l'ordre de pipeline.classes_
    classes = list(pipeline.classes_)
    idx_refus, idx_accord = classes.index(0), classes.index(1)

    return [
        {
            "prediction": int(pred),
            "decision": "CRÉDIT ACCORDÉ" if pred == 1 else "CRÉDIT REFUSÉ",
            "probabilite_refus": float(proba[idx_refus]),
            "probabilite_accord": float(proba[idx_accord]),
            "confiance": float(proba.max())
        }
        for pred, proba in zip(preds, probas)
    ]


@app.get("/")
def home():
    return {
        "message": "API Credit Scoring",
        "status": "operational",
        "endpoints": {
            "/predict/": "Prédire pour 1 client",
            "/predict_batch/": "Prédire pour plusieurs clients",
            "/health/": "État de l'API",
            "/docs": "Documentation"
        }
    }


@app.get("/health/")
def health():
    return {
        "status": "healthy",
        "pipeline_file": PIPELINE_PATH.name,
        "pipeline_type": str(type(pipeline)),
        "pipeline_steps": [step[0] for step in pipeline.steps]
    }


@app.post("/predict/")
def predict(data: CreditRequest):
    """Prédiction pour 1 client"""
    try:
        return predire([data])[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_batch/")
def predict_batch(clients: List[CreditRequest]):
    """Prédiction pour plusieurs clients"""
    if not clients:
        raise HTTPException(status_code=422, detail="La liste de clients est vide")

    try:
        results = predire(clients)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    for i, result in enumerate(results):
        result["client_id"] = i + 1

    total = len(results)
    accordes = sum(1 for r in results if r["prediction"] == 1)

    return {
        "total": total,
        "accordes": accordes,
        "refuses": total - accordes,
        "taux_acceptation": f"{(accordes/total)*100:.1f}%",
        "predictions": results
    }
