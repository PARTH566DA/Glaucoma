# Glaucoma Backend API

This API serves glaucoma prediction for a single fundus image.

## What the API Returns
- Prediction label (`GLAUCOMA`, `GLAUCOMA_SUSPECT`, or `NORMAL`)
- Confidence score
- Extracted features:
  - Disc/Cup/Rim area
  - Disc/Cup width and height
  - ACDR, VCDR, HCDR
- Bounding boxes for disc and cup
- Annotated fundus image as base64 JPEG (`annotated_image_base64`)

## Files Added
- `glaucoma_api.py`: FastAPI service
- `requirements_api.txt`: API dependencies
- `train_glaucoma_model.py`: trains calibrated model with proper labels and anomaly detector

## Model Behavior
The service loads a saved model from `models/glaucoma_rf.joblib`.

- If model exists: uses ML prediction + anomaly detection.
- If model is missing: falls back to rule-based CDR thresholds.

Recommended workflow: always train the model first with real labels, then start the API.

You can override model path with:
- `GLAUCOMA_MODEL_PATH`

## Run Locally

### 1. Install dependencies
```bash
pip install -r requirements_api.txt
```

### 2. Start API
```bash
python3 -m uvicorn glaucoma_api:app --host 0.0.0.0 --port 8000 --reload
```

### Optional but recommended: train model with Mean labels before start
```bash
python3 train_glaucoma_model.py \
  --train-features-csv /Users/dau/Desktop/college/BTP/Dataset/Chakshu/20123135/extracted_features_remidio_train.csv \
  --train-labels-csv /Users/dau/Desktop/college/BTP/Dataset/Chakshu/20123135/Train/6.0_Glaucoma_Decision/Mean/Remidio.csv \
  --output-model models/glaucoma_rf.joblib
```

### 3. Open docs
- Swagger UI: `http://127.0.0.1:8000/docs`

## Test with curl
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/absolute/path/to/fundus.jpg"
```

## Response Example (trimmed)
```json
{
  "filename": "fundus.jpg",
  "prediction": "GLAUCOMA",
  "confidence": 0.8421,
  "model_mode": "ml_model",
  "anomaly_score": -0.012345,
  "anomaly_flag": false,
  "warnings": [],
  "features": {
    "Disc Area": 12543.0,
    "Cup Area": 6742.0,
    "Rim Area": 5801.0,
    "ACDR": 0.5375,
    "VCDR": 0.6522,
    "HCDR": 0.6190
  },
  "annotated_image_base64": "..."
}
```

## Decode annotated image from base64 (Python)
```python
import base64

with open("annotated.jpg", "wb") as f:
    f.write(base64.b64decode(response_json["annotated_image_base64"]))
```

## Notes
- The current segmentation is OpenCV threshold based. Accuracy depends on image quality and optic disc visibility.
- For better clinical performance, replace segmentation with a trained disc/cup segmentation model.
- If `anomaly_flag` is true, treat prediction with caution (input is out-of-distribution relative to training data).
