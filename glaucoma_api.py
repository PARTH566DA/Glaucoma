#!/usr/bin/env python3
"""
Backend API for glaucoma prediction from a single fundus image.

Input:
- One fundus image (multipart file upload)

Output:
- Prediction label
- Confidence score (if ML model is available)
- Extracted cup/disc features
- Annotated image (base64 JPEG)
"""

import base64
import os
from datetime import datetime
from typing import Dict, Optional, Tuple

import cv2
import joblib
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.base import ClassifierMixin
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


FEATURE_COLS = [
    "Disc Area",
    "Cup Area",
    "Rim Area",
    "Cup Height",
    "Cup Width",
    "Disc Height",
    "Disc Width",
    "ACDR",
    "VCDR",
    "HCDR",
]

RAW_TO_STANDARD_FEATURE_MAP = {
    "Disc_Area": "Disc Area",
    "Cup_Area": "Cup Area",
    "Rim_Area": "Rim Area",
    "Cup_Height": "Cup Height",
    "Cup_Width": "Cup Width",
    "Disc_Height": "Disc Height",
    "Disc_Width": "Disc Width",
    "ACDR": "ACDR",
    "VCDR": "VCDR",
    "HCDR": "HCDR",
}

DEFAULT_MODEL_PATH = os.environ.get("GLAUCOMA_MODEL_PATH", "models/glaucoma_rf.joblib")


class HealthResponse(BaseModel):
    status: str
    model_ready: bool
    model_mode: str
    message: str



# --- U-Net Architecture for Inference ---
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_c=3, out_c=1):
        super().__init__()
        self.d1 = DoubleConv(in_c, 32); self.p1 = nn.MaxPool2d(2)
        self.d2 = DoubleConv(32, 64); self.p2 = nn.MaxPool2d(2)
        self.d3 = DoubleConv(64, 128); self.p3 = nn.MaxPool2d(2)
        self.bn = DoubleConv(128, 256)
        self.u1 = nn.ConvTranspose2d(256, 128, 2, stride=2); self.c1 = DoubleConv(256, 128)
        self.u2 = nn.ConvTranspose2d(128, 64, 2, stride=2); self.c2 = DoubleConv(128, 64)
        self.u3 = nn.ConvTranspose2d(64, 32, 2, stride=2); self.c3 = DoubleConv(64, 32)
        self.out = nn.Conv2d(32, out_c, 1)

    def forward(self, x):
        d1 = self.d1(x); p1 = self.p1(d1)
        d2 = self.d2(p1); p2 = self.p2(d2)
        d3 = self.d3(p2); p3 = self.p3(d3)
        bn = self.bn(p3)
        u1 = self.u1(bn); c1 = self.c1(torch.cat([d3, u1], dim=1))
        u2 = self.u2(c1); c2 = self.c2(torch.cat([d2, u2], dim=1))
        u3 = self.u3(c2); c3 = self.c3(torch.cat([d1, u3], dim=1))
        return torch.sigmoid(self.out(c3))

class GlaucomaPredictor:
    """Performs segmentation, feature extraction, model inference, and annotation."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self.model: Optional[ClassifierMixin] = None
        self.scaler: Optional[StandardScaler] = None
        self.anomaly_detector: Optional[IsolationForest] = None
        self.anomaly_threshold: float = -0.10
        self.model_meta: Dict = {}
        self.feature_order = FEATURE_COLS.copy()
        self.model_mode = "rule_based"
        self.normal_confidence_threshold = 0.90

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self._load_model()
        self._load_unets()

    def _load_model(self) -> None:
        if os.path.isfile(self.model_path):
            payload = joblib.load(self.model_path)
            self.model = payload.get("model")
            self.scaler = payload.get("scaler")
            self.feature_order = payload.get("feature_order", FEATURE_COLS.copy())
            self.anomaly_detector = payload.get("anomaly_detector")
            self.anomaly_threshold = float(payload.get("anomaly_threshold", -0.10))
            self.model_meta = {
                "trained_at": payload.get("trained_at"),
                "trained_from_features": payload.get("trained_from_features"),
                "trained_from_labels": payload.get("trained_from_labels"),
                "metrics": payload.get("metrics", {}),
                "label_schema": payload.get("label_schema", "unknown"),
            }
            if self.model is not None and self.scaler is not None:
                self.model_mode = "ml_model"
                return

        # No valid model payload available; keep rule-based mode.
        self.model = None
        self.scaler = None
        self.anomaly_detector = None
        self.model_mode = "rule_based"

    def _load_unets(self) -> None:
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.unet_disc = None
        self.unet_cup = None
        
        disc_path = "models/unet_disc.pth"
        cup_path = "models/unet_cup.pth"
        
        if os.path.exists(disc_path) and os.path.exists(cup_path):
            self.unet_disc = UNet().to(self.device)
            self.unet_disc.load_state_dict(torch.load(disc_path, map_location=self.device))
            self.unet_disc.eval()
            
            self.unet_cup = UNet().to(self.device)
            self.unet_cup.load_state_dict(torch.load(cup_path, map_location=self.device))
            self.unet_cup.eval()

    def _validate_image_for_screening(self, image: np.ndarray) -> Tuple[bool, Optional[str], list[str]]:
        """Validate uploaded image shape and quality before segmentation."""
        h, w = image.shape[:2]
        warnings = []

        min_dim = min(h, w)
        if min_dim < 224:
            return False, "Image is too small for reliable screening. Please upload at least 224x224 pixels.", warnings

        aspect_ratio = max(h, w) / max(1, min(h, w))
        if aspect_ratio > 2.5:
            return False, "Image aspect ratio is unusual for fundus screening. Please upload a centered single-eye fundus image.", warnings

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_intensity = float(np.mean(gray))
        contrast = float(np.std(gray))
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        if mean_intensity < 25 or mean_intensity > 235:
            warnings.append("Image brightness is extreme; results may be unreliable.")
        if contrast < 12:
            warnings.append("Image contrast is very low; segmentation may be unreliable.")
        if blur_score < 20:
            warnings.append("Image appears blurry; segmentation may be unreliable.")

        return True, None, warnings

    @staticmethod
    def _validate_segmentation_geometry(
        disc_area: int,
        cup_area: int,
        disc_width: int,
        disc_height: int,
        cup_width: int,
        cup_height: int,
        image_shape: Tuple[int, int, int],
    ) -> Tuple[bool, Optional[str], list[str]]:
        """Sanity-check segmentation outputs to avoid invalid ratios/markings."""
        h, w = image_shape[:2]
        img_area = max(1, h * w)
        warnings = []

        if disc_area <= 0 or disc_width <= 0 or disc_height <= 0:
            return False, "Could not detect optic disc reliably. Please upload a clear centered fundus image.", warnings
        if cup_area < 0 or cup_width < 0 or cup_height < 0:
            return False, "Invalid cup segmentation detected. Please try another fundus image.", warnings
        if cup_area >= disc_area:
            return False, "Cup segmentation is inconsistent with disc region. Please upload a clearer fundus image.", warnings

        disc_area_ratio = disc_area / img_area
        cup_area_ratio = cup_area / img_area
        if disc_area_ratio < 0.002 or disc_area_ratio > 0.40:
            return False, "Detected optic disc size is outside expected range. Please upload a proper fundus image.", warnings

        if cup_area_ratio < 0.0002:
            warnings.append("Cup region is very small; ratios may be unstable.")

        return True, None, warnings

    def _get_unet_mask(self, image: np.ndarray, model: nn.Module) -> np.ndarray:
        orig_h, orig_w = image.shape[:2]
        img_resized = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (256, 256))
        img_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
        
        with torch.no_grad():
            pred = model(img_tensor).squeeze().cpu().numpy()
            
        mask = (pred > 0.5).astype(np.uint8) * 255
        mask_resized = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        return mask_resized

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return enhanced

    def detect_optic_disc(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        enhanced = self.preprocess_image(image)
        _, bright_regions = cv2.threshold(enhanced, 200, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        bright_regions = cv2.morphologyEx(bright_regions, cv2.MORPH_CLOSE, kernel)
        bright_regions = cv2.morphologyEx(bright_regions, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(bright_regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            h, w = enhanced.shape
            disc_mask = np.zeros_like(enhanced)
            radius = max(8, min(w, h) // 8)
            center = (w // 2, h // 2)
            cv2.circle(disc_mask, center, radius, 255, -1)
            bbox = (center[0] - radius, center[1] - radius, radius * 2, radius * 2)
            return disc_mask, bbox

        largest_contour = max(contours, key=cv2.contourArea)
        disc_mask = np.zeros_like(enhanced)
        cv2.drawContours(disc_mask, [largest_contour], -1, 255, -1)
        bbox = cv2.boundingRect(largest_contour)
        return disc_mask, bbox

    def detect_optic_cup(
        self,
        image: np.ndarray,
        disc_mask: np.ndarray,
        disc_bbox: Tuple[int, int, int, int],
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        x, y, w, h = disc_bbox
        h_img, w_img = image.shape[:2]

        x = max(0, min(x, w_img - 1))
        y = max(0, min(y, h_img - 1))
        w = max(1, min(w, w_img - x))
        h = max(1, min(h, h_img - y))

        disc_region = image[y : y + h, x : x + w].copy()
        disc_mask_region = disc_mask[y : y + h, x : x + w]

        gray = cv2.cvtColor(disc_region, cv2.COLOR_BGR2GRAY)
        _, cup_region = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        cup_region = cv2.morphologyEx(cup_region, cv2.MORPH_CLOSE, kernel)
        cup_region = cv2.morphologyEx(cup_region, cv2.MORPH_OPEN, kernel)

        cup_region = cv2.bitwise_and(cup_region, disc_mask_region)

        cup_mask = np.zeros_like(disc_mask)
        cup_mask[y : y + h, x : x + w] = cup_region

        contours, _ = cv2.findContours(cup_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            cx, cy, cw, ch = cv2.boundingRect(largest)
            cup_bbox = (x + cx, y + cy, cw, ch)
            return cup_mask, cup_bbox

        fallback_w, fallback_h = max(1, int(w * 0.6)), max(1, int(h * 0.6))
        fx = x + (w - fallback_w) // 2
        fy = y + (h - fallback_h) // 2
        center = (x + w // 2, y + h // 2)
        radius = max(1, min(fallback_w, fallback_h) // 2)
        cv2.circle(cup_mask, center, radius, 255, -1)
        return cup_mask, (fx, fy, fallback_w, fallback_h)

    @staticmethod
    def calculate_area(mask: np.ndarray) -> int:
        return int(np.sum(mask > 128))

    @staticmethod
    def _probability_of_normal(probas: np.ndarray, classes: np.ndarray) -> float:
        """Get probability of NORMAL class in a robust way for numeric or string labels."""
        if len(probas) == 0:
            return 0.0

        for idx, cls in enumerate(classes):
            # Numeric schema: 0 is NORMAL in this project.
            if str(cls) == "0":
                return float(probas[idx])
            # String schema fallback.
            if str(cls).strip().upper() == "NORMAL":
                return float(probas[idx])

        # Fallback when class schema is unknown.
        return float(probas[0])

    def _predict_from_features(self, feature_map: Dict[str, float]) -> Tuple[str, float, str, float, bool]:
        if self.model is not None and self.scaler is not None:
            x = np.array([[feature_map[col] for col in self.feature_order]], dtype=float)
            x_scaled = self.scaler.transform(x)

            probas = self.model.predict_proba(x_scaled)[0]
            classes = getattr(self.model, "classes_", np.array([]))

            prob_normal = self._probability_of_normal(probas, classes)
            prob_not_normal = max(0.0, 1.0 - prob_normal)

            # Conservative medical messaging:
            # return NORMAL only when model confidence is high and ratios are comfortably low.
            has_ratio_concern = (
                feature_map["ACDR"] > 0.27
                or feature_map["VCDR"] > 0.45
                or feature_map["HCDR"] > 0.60
            )

            if prob_normal >= self.normal_confidence_threshold and not has_ratio_concern:
                label = "NORMAL"
                final_confidence = prob_normal
            else:
                label = "CONSULT_OPHTHALMOLOGIST"
                final_confidence = max(prob_not_normal, 0.55)

            anomaly_score = 0.0
            anomaly_flag = False
            if self.anomaly_detector is not None:
                anomaly_score = float(self.anomaly_detector.score_samples(x_scaled)[0])
                anomaly_flag = anomaly_score < self.anomaly_threshold
                
            return label, final_confidence, "ml_model", anomaly_score, anomaly_flag

        # Rule-based fallback using strict "normal" thresholds.
        acdr = feature_map["ACDR"]
        vcdr = feature_map["VCDR"]
        hcdr = feature_map["HCDR"]
        if acdr < 0.22 and vcdr < 0.40 and hcdr < 0.55:
            return "NORMAL", 0.80, "rule_based", 0.0, False
        return "CONSULT_OPHTHALMOLOGIST", 0.65, "rule_based", 0.0, False

    def _annotate_image(
        self,
        image: np.ndarray,
        disc_mask: np.ndarray,
        cup_mask: np.ndarray,
        prediction: str,
        confidence: float,
        feature_map: Dict[str, float],
    ) -> np.ndarray:
        overlay = image.copy()

        # Draw translucent masks: disc in green, cup in red.
        overlay[disc_mask > 0] = (0, 180, 0)
        overlay[cup_mask > 0] = (0, 0, 255)
        annotated = cv2.addWeighted(overlay, 0.25, image, 0.75, 0)

        disc_contours, _ = cv2.findContours((disc_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cup_contours, _ = cv2.findContours((cup_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw exact mask boundaries instead of enclosing circles.
        if disc_contours:
            cv2.drawContours(annotated, disc_contours, -1, (0, 255, 0), 3)
        if cup_contours:
            cv2.drawContours(annotated, cup_contours, -1, (0, 0, 255), 3)

        color = (0, 140, 255) if prediction == "CONSULT_OPHTHALMOLOGIST" else (0, 200, 0)
        prediction_text = "CONSULT OPHTHALMOLOGIST" if prediction == "CONSULT_OPHTHALMOLOGIST" else "NORMAL"
        lines = [
            f"Screening Advice: {prediction_text}",
            f"Confidence: {confidence:.3f}",
            f"ACDR: {feature_map['ACDR']:.3f}  VCDR: {feature_map['VCDR']:.3f}  HCDR: {feature_map['HCDR']:.3f}",
        ]

        y = 30
        for idx, line in enumerate(lines):
            line_color = color if idx == 0 else (255, 255, 255)
            cv2.putText(annotated, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(annotated, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, line_color, 2)
            y += 32

        return annotated

    def predict_from_bytes(
        self, 
        image_bytes: bytes, 
        filename: str,
        disc_mask_bytes: Optional[bytes] = None,
        cup_mask_bytes: Optional[bytes] = None
    ) -> Dict:
        np_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Uploaded file is not a valid image.")

        valid_input, input_error, input_warnings = self._validate_image_for_screening(image)
        if not valid_input:
            raise ValueError(input_error)

        if disc_mask_bytes and cup_mask_bytes:
            # Decode provided masks
            d_buffer = np.frombuffer(disc_mask_bytes, dtype=np.uint8)
            disc_mask_img = cv2.imdecode(d_buffer, cv2.IMREAD_GRAYSCALE)
            disc_mask = cv2.resize(disc_mask_img, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            c_buffer = np.frombuffer(cup_mask_bytes, dtype=np.uint8)
            cup_mask_img = cv2.imdecode(c_buffer, cv2.IMREAD_GRAYSCALE)
            cup_mask = cv2.resize(cup_mask_img, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            # Extract bounding boxes from provided masks
            d_contours, _ = cv2.findContours((disc_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            disc_bbox = cv2.boundingRect(max(d_contours, key=cv2.contourArea)) if d_contours else (0,0,0,0)
            
            c_contours, _ = cv2.findContours((cup_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cup_bbox = cv2.boundingRect(max(c_contours, key=cv2.contourArea)) if c_contours else (0,0,0,0)
        else:
            # Use PyTorch U-Net for automatic expert segmentation if available
            if self.unet_disc is not None and self.unet_cup is not None:
                disc_mask = self._get_unet_mask(image, self.unet_disc)
                cup_mask = self._get_unet_mask(image, self.unet_cup)
                
                d_contours, _ = cv2.findContours(disc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                disc_bbox = cv2.boundingRect(max(d_contours, key=cv2.contourArea)) if d_contours else (0,0,0,0)
                
                c_contours, _ = cv2.findContours(cup_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cup_bbox = cv2.boundingRect(max(c_contours, key=cv2.contourArea)) if c_contours else (0,0,0,0)
            else:
                # Fallback to OpenCV heuristic detection if masks aren't provided and U-Net isn't found
                disc_mask, disc_bbox = self.detect_optic_disc(image)
                cup_mask, cup_bbox = self.detect_optic_cup(image, disc_mask, disc_bbox)

        disc_area = self.calculate_area(disc_mask)
        cup_area = self.calculate_area(cup_mask)
        rim_area = max(0, disc_area - cup_area)

        _, _, disc_width, disc_height = disc_bbox
        _, _, cup_width, cup_height = cup_bbox

        valid_seg, seg_error, seg_warnings = self._validate_segmentation_geometry(
            disc_area=disc_area,
            cup_area=cup_area,
            disc_width=disc_width,
            disc_height=disc_height,
            cup_width=cup_width,
            cup_height=cup_height,
            image_shape=image.shape,
        )
        if not valid_seg:
            raise ValueError(seg_error)

        acdr = (cup_area / disc_area) if disc_area > 0 else 0.0
        vcdr = (cup_height / disc_height) if disc_height > 0 else 0.0
        hcdr = (cup_width / disc_width) if disc_width > 0 else 0.0

        feature_map = {
            "Disc Area": float(disc_area),
            "Cup Area": float(cup_area),
            "Rim Area": float(rim_area),
            "Cup Height": float(cup_height),
            "Cup Width": float(cup_width),
            "Disc Height": float(disc_height),
            "Disc Width": float(disc_width),
            "ACDR": float(round(acdr, 4)),
            "VCDR": float(round(vcdr, 4)),
            "HCDR": float(round(hcdr, 4)),
        }

        prediction, confidence, used_mode, anomaly_score, anomaly_flag = self._predict_from_features(feature_map)
        annotated = self._annotate_image(image, disc_mask, cup_mask, prediction, confidence, feature_map)

        success, encoded = cv2.imencode(".jpg", annotated)
        if not success:
            raise RuntimeError("Failed to encode annotated image.")

        annotated_base64 = base64.b64encode(encoded.tobytes()).decode("utf-8")

        warnings = []
        warnings.extend(input_warnings)
        warnings.extend(seg_warnings)
        if anomaly_flag:
            warnings.append("Input feature vector is out-of-distribution relative to model training data.")
        if used_mode == "rule_based":
            warnings.append("Model file not loaded; prediction is rule-based.")
        if prediction == "CONSULT_OPHTHALMOLOGIST":
            warnings.append("Conservative screening policy triggered: follow-up with an ophthalmologist is recommended.")

        if prediction == "NORMAL":
            patient_message = "No high-risk glaucoma pattern detected with high confidence in this screening image."
        else:
            patient_message = "This screening result is not confidently normal. Please consult an ophthalmologist for clinical evaluation."

        return {
            "filename": filename,
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "patient_message": patient_message,
            "model_mode": used_mode,
            "anomaly_score": round(anomaly_score, 6),
            "anomaly_flag": anomaly_flag,
            "warnings": warnings,
            "features": feature_map,
            "disc_bbox": {
                "x": int(disc_bbox[0]),
                "y": int(disc_bbox[1]),
                "width": int(disc_bbox[2]),
                "height": int(disc_bbox[3]),
            },
            "cup_bbox": {
                "x": int(cup_bbox[0]),
                "y": int(cup_bbox[1]),
                "width": int(cup_bbox[2]),
                "height": int(cup_bbox[3]),
            },
            "annotated_image_base64": annotated_base64,
            "annotated_image_format": "jpeg",
            "extracted_at": datetime.utcnow().isoformat() + "Z",
        }


app = FastAPI(
    title="Glaucoma Fundus API",
    description="Upload one fundus image and receive glaucoma prediction, extracted features, and annotated image.",
    version="1.0.0",
)

predictor = GlaucomaPredictor()

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    with open("frontend.html", "r") as f:
        return f.read()

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    if predictor.model_mode == "ml_model":
        message = "ML model is loaded and ready."
        ready = True
    else:
        message = "Running in rule-based mode; train model first with train_glaucoma_model.py."
        ready = False

    return HealthResponse(
        status="ok",
        model_ready=ready,
        model_mode=predictor.model_mode,
        message=message,
    )


@app.get("/model-info")
def model_info() -> Dict:
    return {
        "model_path": predictor.model_path,
        "model_mode": predictor.model_mode,
        "features": predictor.feature_order,
        "anomaly_threshold": predictor.anomaly_threshold,
        "metadata": predictor.model_meta,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...), disc_mask: Optional[UploadFile] = File(None), cup_mask: Optional[UploadFile] = File(None)) -> Dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing file name.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Basic extension validation first; final validation is image decoding.
    allowed = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    ext = os.path.splitext(file.filename.lower())[1]
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Use one of: {sorted(allowed)}",
        )

    try:
        d_bytes = await disc_mask.read() if disc_mask else None
        c_bytes = await cup_mask.read() if cup_mask else None
        result = predictor.predict_from_bytes(content, file.filename, d_bytes, c_bytes)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(exc)}") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("glaucoma_api:app", host="0.0.0.0", port=8000, reload=True)
