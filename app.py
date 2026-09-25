from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH=Path("champion_model.joblib")
if not MODEL_PATH.exists():
    raise RuntimeError("Copy champion_model.joblib from Task 04 into this repository.")
model=joblib.load(MODEL_PATH)

app=FastAPI(title="RabTech Task 06 - ML Inference API",version="1.0.0")

FEATURES=["age","workclass","fnlwgt","education","education-num","marital-status",
          "occupation","relationship","race","sex","capital-gain","capital-loss",
          "hours-per-week","native-country"]

class PredictionRequest(BaseModel):
    age:int=Field(...,ge=17,le=100)
    workclass:str
    fnlwgt:int=Field(...,gt=0)
    education:str
    education_num:int=Field(...,ge=1,le=20)
    marital_status:str
    occupation:str
    relationship:str
    race:str
    sex:str
    capital_gain:int=Field(...,ge=0)
    capital_loss:int=Field(...,ge=0)
    hours_per_week:int=Field(...,ge=1,le=168)
    native_country:str

@app.get("/")
def root():
    return {"service":"RabTech ML Inference API","status":"running","docs":"/docs"}

@app.get("/health")
def health():
    return {"status":"healthy","model_loaded":True}

@app.post("/predict")
def predict(p:PredictionRequest):
    try:
        values=[p.age,p.workclass,p.fnlwgt,p.education,p.education_num,p.marital_status,
                p.occupation,p.relationship,p.race,p.sex,p.capital_gain,p.capital_loss,
                p.hours_per_week,p.native_country]
        X=pd.DataFrame([values],columns=FEATURES)
        prediction=model.predict(X)[0]
        probabilities=None
        if hasattr(model,"predict_proba"):
            probs=model.predict_proba(X)[0]
            probabilities={str(c):round(float(v),6) for c,v in zip(model.classes_,probs)}
        return {"prediction":str(prediction),"probabilities":probabilities}
    except Exception as e:
        raise HTTPException(status_code=422,detail=f"Prediction failed: {e}")
