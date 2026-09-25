# RabTech Academy Task 06 — Real-Time ML Inference REST API & Capstone

This project packages the **Task 04 champion model** as a FastAPI REST service with JSON schema validation, prediction probabilities, automated tests, Docker containerization, and API documentation.

## Architecture

`JSON Client → FastAPI /predict → Pydantic validation → champion_model.joblib → prediction + probabilities → JSON`

## Required model
Copy **champion_model.joblib** from Task 04 into this repository root. The application intentionally will not start without the real trained model.

## Files
- `app.py` — FastAPI server and `/predict`
- `champion_model.joblib` — Task 04 trained model (add this yourself)
- `test_api.py` — unit tests
- `Dockerfile` — container
- `requirements.txt` — pinned dependencies
- `README.md` — documentation

## Run locally
```bash
pip install -r requirements.txt
uvicorn app:app --reload
```
Open `http://127.0.0.1:8000/docs`.

## POST /predict sample
```json
{
  "age": 39,
  "workclass": "State-gov",
  "fnlwgt": 77516,
  "education": "Bachelors",
  "education_num": 13,
  "marital_status": "Never-married",
  "occupation": "Adm-clerical",
  "relationship": "Not-in-family",
  "race": "White",
  "sex": "Male",
  "capital_gain": 2174,
  "capital_loss": 0,
  "hours_per_week": 40,
  "native_country": "United-States"
}
```

Response contains the model's predicted class and class probabilities.

## Tests
```bash
pytest -v
```
Tests check success responses, `/predict` response structure, invalid-value validation, and missing required fields.

## Docker
```bash
docker build -t rabtech-ml-api .
docker run -p 8000:8000 rabtech-ml-api
```

## End-to-end design
The Task 04 training workflow serializes the champion classification pipeline. This API loads that artifact at startup. Pydantic validates incoming JSON, the trained pipeline performs inference, and FastAPI returns the class and probabilities. Docker packages the service and pinned dependencies into a reproducible deployment unit.

**Note:** Do not commit secrets or API keys. This is an educational ML deployment project.
