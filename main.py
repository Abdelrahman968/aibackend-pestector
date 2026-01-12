import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union, Literal
from io import BytesIO
import time
import numpy as np
from PIL import Image, UnidentifiedImageError
import requests
import json
import exifread
import humanize
import re
import shutil
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler

# --- PyTorch Imports ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

# --- TensorFlow Imports ---
from tensorflow.keras.models import load_model as tf_load_model

# --- FastAPI Imports ---
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Depends, status, Query
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security import APIKeyHeader

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s] - %(message)s',
    handlers=[
        logging.FileHandler('app_combined_v2_2_5.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Plant Disease Classifier (ViT + VGG)",
    description="Upload an image to classify plant disease using PyTorch Vision Transformer, TensorFlow VGG, or the best result from both. Optionally uses Gemini 1.5 Flash for summarized details.",
    version="2.2.5-gemini-fix"
)

# --- CORS Middleware ---
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- Static Files ---
static_dir = Path("static")
if static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    logger.info(f"Mounted static directory: {static_dir.resolve()}")
else:
    logger.warning(f"Static directory not found at {static_dir.resolve()}. Frontend ('/') might not work.")

# --- Directory Configurations ---
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# --- API Key Authentication ---
API_KEY_NAME = "X-API-KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
VALID_API_KEYS = {"1122333", "445566"}

async def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key not in VALID_API_KEYS:
        logger.warning(f"Invalid API Key: {api_key}")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API Key")
    return api_key

# --- Common Configuration ---
class CommonConfig:
    CLASS_LABELS = [
        'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
        'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
        'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
        'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
        'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
        'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
        'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
        'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
        'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
        'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
        'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
        'Tomato___healthy'
    ]
    NUM_CLASSES = len(CLASS_LABELS)
    LOW_CONFIDENCE_WARNING_THRESHOLD = 0.70

# --- PyTorch ViT Model Setup ---
class ViTConfig:
    MODEL_PATH = "model/plant_disease_vit_BEST_model_state.pth"
    IMAGE_SIZE = 224
    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD = [0.229, 0.224, 0.225]

def get_pytorch_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

pytorch_device = get_pytorch_device()

class ViTImageClassificationBase(nn.Module):
    pass

class ViTPlantClassifier(ViTImageClassificationBase):
    def __init__(self, num_classes):
        super().__init__()
        logger.debug(f"Initializing ViTPlantClassifier architecture for {num_classes} classes.")
        try:
            self.network = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
            logger.debug("ViT backbone loaded using 'weights' argument.")
        except TypeError:
            logger.warning("ViT 'weights' arg failed. Trying 'pretrained=True'.")
            try:
                self.network = models.vit_b_16(pretrained=True)
                logger.debug("ViT backbone loaded using 'pretrained=True'.")
            except Exception as e_inner:
                logger.error(f"ViT backbone load failed: {e_inner}", exc_info=True)
                raise RuntimeError(f"Could not load ViT backbone: {e_inner}")
        except Exception as e_outer:
            logger.error(f"Fatal error loading ViT backbone: {e_outer}", exc_info=True)
            raise RuntimeError(f"Could not load ViT backbone: {e_outer}")
        try:
            if hasattr(self.network, 'heads') and isinstance(self.network.heads.head, nn.Linear):
                num_ftrs = self.network.heads.head.in_features
                self.network.heads.head = nn.Linear(num_ftrs, num_classes)
                logger.debug(f"Replaced ViT classifier head ({num_ftrs} -> {num_classes}).")
            else:
                logger.error(f"Unexpected ViT structure: {type(getattr(self.network, 'heads', None))}")
                raise AttributeError("Could not find or replace ViT classifier head.")
        except AttributeError as e_attr:
            logger.error(f"Attribute error finding ViT head: {e_attr}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Error replacing ViT classifier head: {e}", exc_info=True)
            raise RuntimeError(f"Could not replace classifier head: {e}")

    def forward(self, xb: torch.Tensor) -> torch.Tensor:
        return self.network(xb)

loaded_vit_model: Optional[ViTPlantClassifier] = None

def load_pytorch_vit_model():
    global loaded_vit_model
    logger.info("Loading PyTorch ViT model...")
    if loaded_vit_model:
        return
    path = Path(ViTConfig.MODEL_PATH)
    logger.info(f"Path: {path.resolve()}")
    if not path.is_file():
        raise FileNotFoundError(f"ViT Model not found: {path.resolve()}")
    try:
        arch = ViTPlantClassifier(CommonConfig.NUM_CLASSES)
        state = torch.load(path, map_location=pytorch_device)
        arch.load_state_dict(state)
        loaded_vit_model = arch.to(pytorch_device)
        loaded_vit_model.eval()
        logger.info("PyTorch ViT loaded.")
    except Exception as e:
        logger.critical(f"FATAL: ViT Load Error: {e}", exc_info=True)
        raise

def preprocess_image_vit(b: bytes) -> torch.Tensor:
    try:
        img = Image.open(BytesIO(b)).convert('RGB')
        tfm = transforms.Compose([
            transforms.Resize((ViTConfig.IMAGE_SIZE, ViTConfig.IMAGE_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=ViTConfig.NORM_MEAN, std=ViTConfig.NORM_STD)
        ])
        return tfm(img).unsqueeze(0)
    except Exception as e:
        logger.error(f"ViT Preprocessing failed: {e}", exc_info=True)
        raise HTTPException(500, f"ViT preprocessing failed: {e}")

@torch.no_grad()
def predict_vit(t: torch.Tensor) -> Dict[str, Any]:
    if not loaded_vit_model:
        raise RuntimeError("ViT Model not loaded.")
    loaded_vit_model.eval()
    try:
        probs = F.softmax(loaded_vit_model(t.to(pytorch_device)), dim=1)[0]
        top_p, top_i = torch.topk(probs, 3)
        top_3 = [
            {"class": CommonConfig.CLASS_LABELS[i.item()], "confidence": p.item()}
            for i, p in zip(top_i, top_p) if 0 <= i.item() < CommonConfig.NUM_CLASSES
        ]
        best = top_3[0] if top_3 else {"class": "Unknown", "confidence": 0.0}
        return {
            "top_prediction_label": best['class'],
            "confidence": best['confidence'],
            "top_3": top_3
        }
    except Exception as e:
        logger.error(f"ViT Prediction failed: {e}", exc_info=True)
        raise RuntimeError(f"ViT Prediction failed: {e}")

# --- TensorFlow VGG Model Setup ---
class TFVGGConfig:
    MODEL_PATH = "model/vgg_model.h5"
    IMAGE_SIZE = (224, 224)

loaded_tf_model = None

def load_tensorflow_vgg_model():
    global loaded_tf_model
    logger.info("Loading TensorFlow VGG model...")
    if loaded_tf_model:
        return
    path = Path(TFVGGConfig.MODEL_PATH)
    logger.info(f"Path: {path.resolve()}")
    if not path.is_file():
        raise FileNotFoundError(f"TF Model not found: {path.resolve()}")
    try:
        loaded_tf_model = tf_load_model(path)
        assert loaded_tf_model.output_shape[-1] == CommonConfig.NUM_CLASSES, "TF Output shape mismatch"
        logger.info("TensorFlow VGG loaded.")
    except Exception as e:
        logger.critical(f"FATAL: TF Load Error: {e}", exc_info=True)
        raise

def preprocess_image_tf(b: bytes) -> Tuple[np.ndarray, Dict[str, Any]]:
    try:
        img_io = BytesIO(b)
        img = Image.open(img_io)
        meta = {
            "size": humanize.naturalsize(len(b)),
            "format": img.format,
            "width": img.width,
            "height": img.height,
            "resolution": f"{img.width}x{img.height}",
            "mode": img.mode,
            "has_exif": False,
            "has_geo": False,
            "datetime": None,
            "make": None,
            "model": None
        }
        try:
            img_io.seek(0)
            exif = exifread.process_file(img_io, stop_tag='DateTimeOriginal')
            meta.update({
                "has_exif": bool(exif),
                "datetime": str(exif.get('EXIF DateTimeOriginal')),
                "make": str(exif.get('Image Make')),
                "model": str(exif.get('Image Model')),
                "has_geo": any(k.startswith('GPS') for k in exif)
            })
        except Exception as e:
            logger.warning(f"EXIF error: {e}")
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize(TFVGGConfig.IMAGE_SIZE, Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.
        arr = np.expand_dims(arr, axis=0)
        return arr, meta
    except Exception as e:
        logger.error(f"TF Preprocessing failed: {e}", exc_info=True)
        raise HTTPException(500, f"TF preprocessing failed: {e}")

def predict_tf(arr: np.ndarray) -> Dict[str, Any]:
    if not loaded_tf_model:
        raise RuntimeError("TF Model not loaded.")
    try:
        preds = loaded_tf_model.predict(arr, verbose=0)[0]
        top_i = np.argsort(preds)[-3:][::-1]
        top_3 = [
            {"class": CommonConfig.CLASS_LABELS[i], "confidence": float(preds[i])}
            for i in top_i if 0 <= i < CommonConfig.NUM_CLASSES
        ]
        best = top_3[0] if top_3 else {"class": "Unknown", "confidence": 0.0}
        return {
            "top_prediction_label": best['class'],
            "confidence": best['confidence'],
            "top_3": top_3
        }
    except Exception as e:
        logger.error(f"TF Prediction failed: {e}", exc_info=True)
        raise RuntimeError(f"TF Prediction failed: {e}")

# --- Disease Info & Gemini ---
DISEASE_PATTERNS = {
    "blight": {"description": "Causes brown lesions", "severity": "high"},
    "spot": {"description": "Creates spotted patterns", "severity": "medium"},
    "rust": {"description": "Produces rusty spots", "severity": "medium"},
    "virus": {"description": "Causes deformation/discoloration", "severity": "high"},
    "mold": {"description": "Creates fuzzy growth", "severity": "medium"}
}

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "######################")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

def get_disease_info(dn: str):
    dl = dn.lower()
    mtch = [p for p, i in DISEASE_PATTERNS.items() if p in dl]
    dsc = [DISEASE_PATTERNS[p]['description'] for p in mtch]
    sev = "low"
    if any(DISEASE_PATTERNS[p]['severity'] == 'high' for p in mtch):
        sev = "high"
    elif any(DISEASE_PATTERNS[p]['severity'] == 'medium' for p in mtch):
        sev = "medium"
    return {"patterns": mtch, "descriptions": dsc, "severity": sev}

def get_gemini_response(prompt: str) -> Dict[str, Any]:
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        logger.error("Gemini API key not configured.")
        return {"text": "Gemini API key not configured.", "success": False}
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 100}
    }
    try:
        response = requests.post(
            f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
            headers=headers,
            json=data,
            timeout=10
        )
        response.raise_for_status()
        result = response.json()
        text = result["candidates"][0]["content"]["parts"][0]["text"].strip()
        logger.info(f"Gemini response received: {text[:50]}...")
        return {"text": text, "success": True}
    except requests.exceptions.HTTPError as e:
        if response.status_code == 401:
            logger.error("Gemini API key invalid or unauthorized.")
            return {"text": "Invalid Gemini API key.", "success": False}
        logger.error(f"Gemini HTTP error: {e}")
        return {"text": f"Gemini HTTP error: {e}", "success": False}
    except requests.exceptions.Timeout:
        logger.error("Gemini request timed out.")
        return {"text": "Gemini request timed out.", "success": False}
    except requests.exceptions.RequestException as e:
        logger.error(f"Gemini request failed: {e}")
        return {"text": f"Gemini request failed: {e}", "success": False}
    except (KeyError, IndexError) as e:
        logger.error(f"Gemini response parsing error: {e}")
        return {"text": "Error parsing Gemini response.", "success": False}
    except Exception as e:
        logger.error(f"Unexpected Gemini error: {e}", exc_info=True)
        return {"text": f"Unexpected Gemini error: {e}", "success": False}

def extract_first_sentence(t: str) -> str:
    if not t or not isinstance(t, str):
        return ""
    t = t.strip()
    if not t:
        return ""
    m = re.search(r'([.!?])(?:\s+|$)', t)
    if m:
        fs = t[:m.end()].strip()
        return fs if len(fs) > 1 else t.split('\n')[0]
    return t.split('\n')[0]

# --- FastAPI Events ---
@app.on_event("startup")
async def startup_event():
    logger.info("Startup: Loading models...")
    errs = []
    try:
        load_pytorch_vit_model()
    except Exception as e:
        logger.critical(f"ViT Load Err: {e}", exc_info=True)
        errs.append("ViT")
    try:
        load_tensorflow_vgg_model()
    except Exception as e:
        logger.critical(f"TF Load Err: {e}", exc_info=True)
        errs.append("TF")
    if not errs:
        logger.info("Model loading attempt finished.")
    else:
        logger.critical(f"Startup Fail - errors: {', '.join(errs)}")

# --- API Endpoints ---
@app.get("/health", tags=["System"])
async def health_check():
    vit = loaded_vit_model is not None
    tf = loaded_tf_model is not None
    ok = vit and tf
    return JSONResponse(
        status_code=200 if ok else 503,
        content={
            "status": "healthy" if ok else "unhealthy",
            "models_loaded": {"pytorch_vit": vit, "tensorflow_vgg": tf},
            "pytorch_device": str(pytorch_device),
            "timestamp": datetime.now().isoformat()
        }
    )

def save_upload(file: UploadFile) -> Path:
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    ext = Path(file.filename).suffix if file.filename else ".img"
    safe_fn = f"upload_{ts}_{os.urandom(4).hex()}{ext}"
    path = UPLOAD_DIR / safe_fn
    try:
        with path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"File saved: {path}")
        return path
    except Exception as e:
        logger.error(f"Save failed {path}: {e}", exc_info=True)
        if path.exists():
            try:
                path.unlink()
            except OSError:
                pass
        raise HTTPException(500, "Failed to save file.")

@app.post("/classify", tags=["Classification"])
async def classify_image_combined(
    request: Request,
    file: UploadFile = File(...),
    model_choice: Literal["vit", "vgg", "best"] = Query("best"),
    use_gemini: bool = Query(False),
    api_key: str = Depends(get_api_key)
):
    start = time.time()
    logger.info(f"Request: {file.filename}, Model: {model_choice}, Gemini: {use_gemini}")
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(415, "Unsupported file type.")
    if not file.filename:
        raise HTTPException(400, "Filename missing.")
    try:
        img_bytes = await file.read()
        await file.seek(0)
        if not img_bytes:
            raise HTTPException(400, "Empty file.")
        save_path = save_upload(file)
        vit_res, tf_res, meta = None, None, None
        if model_choice in ["vit", "best"]:
            try:
                vit_in = preprocess_image_vit(img_bytes)
                vit_res = predict_vit(vit_in)
            except Exception as e:
                logger.error(f"ViT Err: {e}")
            if model_choice == "vit" and not meta:
                try:
                    img = Image.open(BytesIO(img_bytes))
                    meta = {"size": humanize.naturalsize(len(img_bytes)), "format": img.format}
                except:
                    meta = {}
        if model_choice in ["vgg", "best"]:
            try:
                tf_in, tf_meta = preprocess_image_tf(img_bytes)
                tf_res = predict_tf(tf_in)
                meta = tf_meta
            except Exception as e:
                logger.error(f"TF Err: {e}")
        best_pred, model_src = None, "N/A"
        if model_choice == "vit" and vit_res:
            best_pred, model_src = vit_res, "PyTorch ViT"
        elif model_choice == "vgg" and tf_res:
            best_pred, model_src = tf_res, "TensorFlow VGG"
        elif model_choice == "best":
            if vit_res and tf_res:
                best_pred, model_src = (vit_res, "ViT") if vit_res['confidence'] >= tf_res['confidence'] else (tf_res, "VGG")
            elif vit_res:
                best_pred, model_src = vit_res, "PyTorch ViT"
            elif tf_res:
                best_pred, model_src = tf_res, "TensorFlow VGG"
        if not best_pred:
            raise HTTPException(500, "Prediction failed.")
        best_lbl = best_pred['top_prediction_label']
        # Normalize plant and condition names
        if '___' in best_lbl:
            plnt, cond = best_lbl.split('___')
            plnt = plnt.replace('_', ' ').title()
            cond = cond.replace('_', ' ').title()
        else:
            plnt, cond = "Unknown", best_lbl.replace('_', ' ').title()
        dis_info = get_disease_info(cond)
        treatment, reason, data_src = "N/A", "N/A", "Local"
        gemini_highlighted = False
        try:
            with open("treatment_recommendations.json", "r", encoding='utf-8') as f:
                treat_data = json.load(f)
                treatment = treat_data.get(best_lbl, "No local treatment data.")
        except FileNotFoundError:
            logger.warning("treatments.json not found.")
            treatment = "Treatment data missing."
        except json.JSONDecodeError:
            logger.warning("treatments.json decode error.")
            treatment = "Treatment data error."
        except Exception as e:
            logger.warning(f"Treatment load error: {e}")
            treatment = "Error loading treatment data."
        try:
            with open("reason.json", "r", encoding='utf-8') as f:
                reason_data = json.load(f)
                reason = reason_data.get(best_lbl, "No local reason data.")
        except FileNotFoundError:
            logger.warning("reason.json not found.")
            reason = "Reason data missing."
        except json.JSONDecodeError:
            logger.warning("reason.json decode error.")
            reason = "Reason data error."
        except Exception as e:
            logger.warning(f"Reason load error: {e}")
            reason = "Error loading reason data."
        if use_gemini:
            gem_start = time.time()
            logger.info("Requesting Gemini (1.5 Flash)...")
            treat_prompt = f"One sentence summary of treatment for {cond} in {plnt} plants."
            reason_prompt = f"One sentence summary of causes for {cond} in {plnt} plants."
            gem_t_res = get_gemini_response(treat_prompt)
            gem_r_res = get_gemini_response(reason_prompt)
            t_ok = gem_t_res["success"]
            r_ok = gem_r_res["success"]
            if t_ok:
                treatment = extract_first_sentence(gem_t_res["text"])
                data_src = "Gemini"
                gemini_highlighted = True
            if r_ok:
                reason = extract_first_sentence(gem_r_res["text"])
                data_src = "Gemini"
                gemini_highlighted = True
            logger.info(f"Gemini took {time.time()-gem_start:.3f}s. Valid: T={t_ok}, R={r_ok}")
        elapsed = time.time() - start
        response = {
            "overall_best_prediction": {
                "plant": plnt,
                "condition": cond,
                "confidence": best_pred['confidence'],
                "confidence_percent": float(f"{best_pred['confidence']*100:.2f}"),
                "model_source": model_src,
                "label": best_lbl,
                "disease_info": dis_info,
                "treatment_recommendations": treatment,
                "reason_for_disease": reason,
                "data_source": data_src,
                "gemini_highlighted": gemini_highlighted
            },
            "vit_predictions": vit_res['top_3'] if vit_res else [],
            "tf_predictions": tf_res['top_3'] if tf_res else [],
            "metadata": {
                "filename": file.filename,
                "content_type": file.content_type,
                "save_path": str(save_path),
                "timestamp": datetime.now().isoformat(),
                "image_details": meta or {}
            },
            "processing_time_seconds": float(f"{elapsed:.3f}"),
            "model_choice_used": model_choice,
            "low_confidence_threshold": CommonConfig.LOW_CONFIDENCE_WARNING_THRESHOLD
        }
        logger.info(f"Success: {file.filename} ({model_choice}) -> {best_lbl} ({best_pred['confidence']:.2%}) in {elapsed:.3f}s")
        return JSONResponse(content=response, headers={"Cache-Control": "no-cache"})
    except HTTPException as e:
        logger.warning(f"HTTP Exc: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(500, f"Internal error: {e}")

# --- File Cleanup Scheduler ---
def cleanup_old_files():
    cutoff = datetime.now() - timedelta(days=7)
    logger.info(f"Cleanup job older than {cutoff.isoformat()}")
    count = 0
    for dp in [UPLOAD_DIR]:
        if not dp.is_dir():
            continue
        for item in dp.iterdir():
            if item.is_file():
                try:
                    if datetime.fromtimestamp(item.stat().st_mtime) < cutoff:
                        item.unlink()
                        logger.info(f"Deleted: {item}")
                        count += 1
                except Exception as e:
                    logger.error(f"Delete failed {item}: {e}")
    logger.info(f"Cleanup finished. Deleted {count} files.")

scheduler = BackgroundScheduler(daemon=True)
scheduler.add_job(cleanup_old_files, 'interval', days=1)
scheduler.start()
logger.info("File cleanup scheduler started.")

@app.on_event("shutdown")
def shutdown_event():
    logger.info("Shutting down scheduler...")
    scheduler.shutdown()

# --- Other Endpoints ---
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def get_index_page():
    path = static_dir / "index.html"
    return HTMLResponse(content=path.read_text(encoding='utf-8')) if path.is_file() else HTMLResponse("Frontend not found.", 404)

@app.delete("/delete-all-files")
async def delete_all_files():
    try:
        def delete_files_in_directory(directory: Path):
            deleted_files = []
            if directory.exists():
                for file_path in directory.iterdir():
                    if file_path.is_file():
                        try:
                            file_path.unlink()
                            deleted_files.append(file_path.name)
                        except Exception as e:
                            raise HTTPException(
                                status_code=500,
                                detail=f"Failed to delete file {file_path.name}: {str(e)}"
                            )
            return deleted_files
        deleted_images = delete_files_in_directory(UPLOAD_DIR)
        return {
            "message": "All files deleted successfully.",
            "deleted_images": deleted_images,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting files: {str(e)}"
        )

@app.get("/health-check")
async def health_check():
    return {"status": "ok"}

@app.get("/scan", response_class=HTMLResponse)
async def get_scan():
    with open("static/scan.html", "r") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content)

@app.get("/changelog", response_class=HTMLResponse)
async def get_changelog():
    with open("static/changelog.html", "r") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content)

@app.get("/doc", response_class=HTMLResponse)
async def get_doc():
    with open("static/doc.html", "r", encoding="utf-8") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content)

@app.get("/library", response_class=HTMLResponse)
async def get_library():
    with open("static/disease-library.html", "r", encoding="utf-8") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content)

@app.get("/treatment", response_class=HTMLResponse)
async def get_treatment():
    with open("static/treatment.html", "r", encoding="utf-8") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content)

@app.get("/research", response_class=HTMLResponse)
async def get_research():
    with open("static/research-papers.html", "r", encoding="utf-8") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content)

@app.get("/pricing", response_class=HTMLResponse)
async def get_pricing():
    with open("static/pricing.html", "r", encoding="utf-8") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content)

@app.get("/contact", response_class=HTMLResponse)
async def get_contact():
    with open("static/contact.html", "r", encoding="utf-8") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content)

@app.get("/faqs", response_class=HTMLResponse)
async def get_faqs():
    with open("static/faqs.html", "r", encoding="utf-8") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content)

@app.get("/knowledge", response_class=HTMLResponse)
async def get_knowledge():
    with open("static/knowledge-base.html", "r", encoding="utf-8") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
# --- END OF main.py ---
