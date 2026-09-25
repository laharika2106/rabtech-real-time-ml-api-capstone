from fastapi.testclient import TestClient
from app import app
client=TestClient(app)
VALID={"age":39,"workclass":"State-gov","fnlwgt":77516,"education":"Bachelors",
"education_num":13,"marital_status":"Never-married","occupation":"Adm-clerical",
"relationship":"Not-in-family","race":"White","sex":"Male","capital_gain":2174,
"capital_loss":0,"hours_per_week":40,"native_country":"United-States"}

def test_root():
    r=client.get("/")
    assert r.status_code==200 and r.json()["status"]=="running"

def test_health():
    r=client.get("/health")
    assert r.status_code==200 and r.json()["model_loaded"] is True

def test_predict_schema_and_response():
    r=client.post("/predict",json=VALID)
    assert r.status_code==200
    assert "prediction" in r.json() and "probabilities" in r.json()

def test_invalid_age():
    x=VALID.copy(); x["age"]=5
    assert client.post("/predict",json=x).status_code==422

def test_missing_field():
    x=VALID.copy(); x.pop("education")
    assert client.post("/predict",json=x).status_code==422
