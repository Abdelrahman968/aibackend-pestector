from fastapi import FastAPI, File, UploadFile, HTTPException,APIRouter, Request, Depends, status
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security import APIKeyHeader
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Union
from io import BytesIO
import requests
import json
import uuid
import exifread
import humanize
from typing import Dict, Any, List, Union, Tuple
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportImage, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# FastAPI app initialization
app = FastAPI(
    title="Plant Disease Classifier",
    description="AI-powered plant disease classification system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# API Key Authentication
API_KEY_NAME = "X-API-KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# List of valid API keys (in production, store these securely)
VALID_API_KEYS = {"1122333", "445566"}

async def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API Key"
        )
    return api_key

class ModelConfig:
    MODEL_PATH = "model/vgg_model.h5"  
    IMAGE_SIZE = (224, 224)
    NUM_CLASSES = 38
    CONFIDENCE_THRESHOLD = 0.85

class PlantDiseaseClassifier:
    _instance = None
    _model = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(PlantDiseaseClassifier, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        self.labels = [
            'Apple___Apple_scab',
            'Apple___Black_rot',
            'Apple___Cedar_apple_rust',
            'Apple___healthy',
            'Blueberry___healthy',
            'Cherry_(including_sour)___Powdery_mildew',
            'Cherry_(including_sour)___healthy',
            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
            'Corn_(maize)___Common_rust_',
            'Corn_(maize)___Northern_Leaf_Blight',
            'Corn_(maize)___healthy',
            'Grape___Black_rot',
            'Grape___Esca_(Black_Measles)',
            'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
            'Grape___healthy',
            'Orange___Haunglongbing_(Citrus_greening)',
            'Peach___Bacterial_spot',
            'Peach___healthy',
            'Pepper,_bell___Bacterial_spot',
            'Pepper,_bell___healthy',
            'Potato___Early_blight',
            'Potato___Late_blight',
            'Potato___healthy',
            'Raspberry___healthy',
            'Soybean___healthy',
            'Squash___Powdery_mildew',
            'Strawberry___Leaf_scorch',
            'Strawberry___healthy',
            'Tomato___Bacterial_spot',
            'Tomato___Early_blight',
            'Tomato___Late_blight',
            'Tomato___Leaf_Mold',
            'Tomato___Septoria_leaf_spot',
            'Tomato___Spider_mites Two-spotted_spider_mite',
            'Tomato___Target_Spot',
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
            'Tomato___Tomato_mosaic_virus',
            'Tomato___healthy'
        ]
        self.categories = self.organize_categories()
        self.disease_patterns = {
            "blight": {
                "keywords": ["blight", "brown", "lesions"],
                "description": "Causes brown lesions and tissue death",
                "severity": "high"
            },
            "spot": {
                "keywords": ["spot", "leaf spot", "lesions"],
                "description": "Creates spotted patterns on leaves",
                "severity": "medium"
            },
            "rust": {
                "keywords": ["rust", "brown spots", "orange"],
                "description": "Produces rusty colored spots",
                "severity": "medium"
            },
            "virus": {
                "keywords": ["virus", "mosaic", "curl", "yellow"],
                "description": "Causes leaf deformation and discoloration",
                "severity": "high"
            },
            "mold": {
                "keywords": ["mold", "fuzzy", "growth"],
                "description": "Creates fuzzy growth on plant surfaces",
                "severity": "medium"
            }
        }
        
        if PlantDiseaseClassifier._model is None:
            try:
                if not os.path.exists(ModelConfig.MODEL_PATH):
                    raise FileNotFoundError(f"Model file not found: {ModelConfig.MODEL_PATH}")
                
                PlantDiseaseClassifier._model = load_model(ModelConfig.MODEL_PATH)
                
                # Verify model output shape
                if PlantDiseaseClassifier._model.output_shape[-1] != ModelConfig.NUM_CLASSES:
                    raise ValueError(
                        f"Model output classes ({PlantDiseaseClassifier._model.output_shape[-1]}) "
                        f"doesn't match expected number of classes "
                        f"({ModelConfig.NUM_CLASSES})"
                    )
                
                logger.info("Model initialized successfully")
                
            except Exception as e:
                logger.error(f"Model initialization error: {str(e)}")
                raise Exception(f"Failed to initialize model: {str(e)}")

    @property
    def model(self):
        return PlantDiseaseClassifier._model

    def organize_categories(self) -> Dict[str, Dict[str, Any]]:
        """Organize plant categories and their diseases."""
        categories = {}
        
        for label in self.labels:
            plant, condition = label.split('___')
            plant = plant.replace('_', ' ')
            
            if plant not in categories:
                categories[plant] = {
                    'healthy': False,
                    'diseases': set(),
                    'total_samples': 0
                }
            
            if condition.lower() == 'healthy':
                categories[plant]['healthy'] = True
            else:
                categories[plant]['diseases'].add(condition.replace('_', ' '))
                
            categories[plant]['total_samples'] += 1
            
        return categories

    def get_disease_info(self, disease_name: str) -> Dict[str, Any]:
        """Analyze disease patterns and get detailed information."""
        disease_lower = disease_name.lower()
        matched_patterns = []
        descriptions = []
        severity = "low"
        
        for pattern, info in self.disease_patterns.items():
            if any(keyword in disease_lower for keyword in info["keywords"]):
                matched_patterns.append(pattern)
                descriptions.append(info["description"])
                if info["severity"] == "high":
                    severity = "high"
                elif info["severity"] == "medium" and severity != "high":
                    severity = "medium"
        
        return {
            "patterns": matched_patterns,
            "descriptions": descriptions,
            "severity": severity
        }

    def predict(self, image_array: np.ndarray) -> dict:
        """Make prediction using the loaded model."""
        try:
            # Get predictions
            predictions = self.model.predict(image_array, verbose=0)
            logger.info(f"Raw predictions shape: {predictions.shape}")  # Log raw predictions shape for debugging
            
            confidence = float(np.max(predictions))
            predicted_idx = np.argmax(predictions)
            
            # Log the predicted index and confidence
            logger.info(f"Predicted index: {predicted_idx}, Confidence: {confidence:.2f}")

            # Get top 3 predictions
            top_3_indices = np.argsort(predictions[0])[-3:][::-1]
            top_3_predictions = []
            
            for idx in top_3_indices:
                class_name = self.labels[idx]
                disease_info = self.get_disease_info(class_name)
                
                prediction_info = {
                    "class": class_name,
                    "confidence": float(predictions[0][idx]),
                    "disease_info": disease_info
                }
                top_3_predictions.append(prediction_info)
            
            return {
                "top_prediction": self.labels[predicted_idx],
                "confidence": confidence,
                "top_3": top_3_predictions
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Prediction failed: {str(e)}"
            )

# Directory configurations
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

# Gemini API configuration
GEMINI_API_KEY = "AIzaSyDxqEMB8Qoq84jn_uSDepoKYnsbwvB5jdc"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"

def get_gemini_response(prompt: str) -> str:
    """Get response from Gemini API."""
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "contents": [{
            "parts": [{
                "text": prompt
            }]
        }]
    }
    response = requests.post(
        f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
        headers=headers,
        json=data
    )
    if response.status_code == 200:
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    else:
        logger.error(f"Gemini API error: {response.status_code}, {response.text}")
        return "No response from Gemini API."

def save_upload(file: UploadFile) -> Path:
    """Save uploaded file and return path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_extension = Path(file.filename).suffix
    save_path = UPLOAD_DIR / f"upload_{timestamp}{file_extension}"
    
    with save_path.open("wb") as buffer:
        file.file.seek(0)
        buffer.write(file.file.read())
    
    return save_path

def preprocess_image(image_bytes: bytes) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Preprocess image to match the model's input requirements and extract metadata.
    Returns a tuple of (preprocessed image array, metadata dictionary)
    """
    try:
        # Load and preprocess image
        img_io = BytesIO(image_bytes)
        img = Image.open(img_io)
        
        # Extract metadata
        metadata = {
            "width": img.width,
            "height": img.height,
            "resolution": f"{img.width} × {img.height}",  # Use proper multiplication symbol
            "format": img.format,
            "mode": img.mode,
            "size": humanize.naturalsize(len(image_bytes)),
            "has_exif": False,
            "has_geo": False,
            "datetime": None,
            "make": None,
            "model": None,
        }
        
        # Extract EXIF data if available
        try:
            img_io.seek(0)  # Reset file pointer
            exif_tags = exifread.process_file(img_io)
            
            if exif_tags:
                metadata["has_exif"] = True
                
                # Extract common EXIF tags
                if 'EXIF DateTimeOriginal' in exif_tags:
                    metadata["datetime"] = str(exif_tags['EXIF DateTimeOriginal'])
                
                if 'Image Make' in exif_tags:
                    metadata["make"] = str(exif_tags['Image Make'])
                
                if 'Image Model' in exif_tags:
                    metadata["model"] = str(exif_tags['Image Model'])
                
                # Check for GPS data
                gps_keys = [key for key in exif_tags.keys() if key.startswith('GPS')]
                if gps_keys:
                    metadata["has_geo"] = True
        except Exception as e:
            logger.warning(f"Error extracting EXIF data: {str(e)}")
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize image
        img = img.resize(ModelConfig.IMAGE_SIZE, Image.LANCZOS)
        
        # Convert to numpy array and normalize
        img_array = np.array(img, dtype=np.float32)
        img_array = img_array / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array, metadata
        
    except Exception as e:
        logger.error(f"Image preprocessing error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process image: {str(e)}"
        )

def generate_pdf_report(report_data: Dict[str, Any], image_path: Path) -> Path:
    """
    Generate a professional PDF report for the plant disease classification with enhanced styling.
    The first page includes the logo, project information, model details, and importance of plants for humans.
    
    Args:
        report_data: Dictionary containing classification results and related information
        image_path: Path to the analyzed plant image
        
    Returns:
        Path: Path to the generated PDF report
    """
    # Create a unique filename for the report
    report_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"plant_report_{report_id}_{timestamp}.pdf"
    report_path = REPORTS_DIR / report_filename
    
    # Create PDF document with margins
    doc = SimpleDocTemplate(
        str(report_path),
        pagesize=letter,
        leftMargin=0.75*inch,
        rightMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )
    styles = getSampleStyleSheet()
    
    # Enhanced custom styles
    custom_styles = {
        'Title': ParagraphStyle(
            'Title',
            parent=styles['Heading1'],
            fontSize=28,
            textColor=colors.HexColor('#1B5E20'),  # Dark green
            spaceAfter=16,
            alignment=1,
            fontName='Helvetica-Bold',
            leading=32
        ),
        'Subtitle': ParagraphStyle(
            'Subtitle',
            parent=styles['Heading2'],
            fontSize=20,
            textColor=colors.HexColor('#1565C0'),  # Dark blue
            spaceAfter=12,
            alignment=1,
            fontName='Helvetica-Bold',
            leading=24
        ),
        'SectionHeader': ParagraphStyle(
            'SectionHeader',
            parent=styles['Heading3'],
            fontSize=16,
            textColor=colors.HexColor('#2E7D32'),  # Medium green
            spaceBefore=12,
            spaceAfter=8,
            fontName='Helvetica-Bold',
            leading=20,
            borderWidth=1,
            borderColor=colors.HexColor('#E8F5E9'),  # Light green
            borderPadding=6,
            borderRadius=4
        ),
        'Normal': ParagraphStyle(
            'Normal',
            parent=styles['Normal'],
            fontSize=11,
            textColor=colors.HexColor('#212121'),  # Dark gray
            spaceBefore=6,
            spaceAfter=6,
            leading=14,
            alignment=0
        ),
        'BulletPoint': ParagraphStyle(
            'BulletPoint',
            parent=styles['Normal'],
            fontSize=11,
            textColor=colors.HexColor('#424242'),
            leftIndent=20,
            spaceBefore=2,
            spaceAfter=2,
            leading=14,
            bulletIndent=10,
            alignment=0
        ),
        'Footer': ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.HexColor('#757575'),  # Medium gray
            alignment=1,
            leading=10
        )
    }
    
    # Prepare content
    content = []
    
    # Add watermark
    def add_watermark(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(colors.HexColor('#E8F5E9'))
        canvas.setFont('Helvetica-Bold', 60)
        canvas.rotate(45)
        canvas.drawString(200, 50, "CONFIDENTIAL")
        canvas.restoreState()
    
    # Add cover page with logo and project information
    content.append(Spacer(1, 1*inch))
    
    # Add logo
    logo_path = "static/logo.png"
    if os.path.exists(logo_path):
        logo = ReportImage(logo_path, width=1.5*inch, height=1.5*inch)
        content.append(logo)
        content.append(Spacer(1, 0.5*inch))
    
    # Add project title
    content.append(Paragraph("Pestector", custom_styles['Title']))
    content.append(Spacer(1, 0.25*inch))
    content.append(Paragraph("Plant Disease Classification System", custom_styles['Subtitle']))
    content.append(Spacer(1, 0.5*inch))
    
    # Add project description
    project_description = """
        <b>Pestector</b> is an AI-powered system designed to identify plant diseases and provide 
        actionable treatment recommendations. Our mission is to help farmers and gardeners 
        maintain healthy crops and reduce losses due to plant diseases.
    """
    content.append(Paragraph(project_description, custom_styles['Normal']))
    content.append(Spacer(1, 0.5*inch))
    
    # Add model information
    model_info = """
        <b>Model Used:</b> VGG16 (Fine-tuned for plant disease classification)
        <br/>
        <b>Accuracy:</b> 98.6% (Tested on PlantVillage dataset)
        <br/>
        <b>Training Data:</b> 38 classes, 54,305 images
    """
    content.append(Paragraph(model_info, custom_styles['Normal']))
    content.append(Spacer(1, 0.5*inch))
    
    # Add importance of plants for humans
    importance_of_plants = """
        <b>Why Plants Matter:</b>
        <br/>
        Plants are essential for human survival. They provide food, oxygen, and raw materials 
        for medicines, clothing, and shelter. Protecting plants from diseases ensures food 
        security and environmental sustainability.
    """
    content.append(Paragraph(importance_of_plants, custom_styles['Normal']))
    content.append(PageBreak())
    
    # Add analyzed image with border and caption
    try:
        img = ReportImage(str(image_path), width=3*inch, height=3*inch)
        img_container = Table(
            [[img]],
            style=TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('TOPPADDING', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
                ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#E0E0E0'))
            ])
        )
        content.append(img_container)
        content.append(Paragraph("Analyzed Plant Image", custom_styles['Normal']))
        content.append(Spacer(1, 0.25*inch))
    except Exception as e:
        logger.error(f"Failed to add image to PDF: {str(e)}")
        content.append(Paragraph("(Image not available)", custom_styles['Normal']))
    
    # Add classification results with enhanced table
    content.append(Paragraph("Classification Results", custom_styles['SectionHeader']))
    
    prediction = report_data["prediction"]
    results_data = [
        ["Plant Species:", prediction["plant"]],
        ["Condition:", prediction["condition"]],
        ["Confidence:", f"{prediction['confidence']}% ({prediction['confidence_level']})"],
        ["Severity Level:", prediction["disease_info"]["severity"].capitalize()]
    ]
    
    results_table = Table(
        results_data,
        colWidths=[1.5*inch, 4*inch],
        style=TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#E8F5E9')),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#1B5E20')),
            ('TEXTCOLOR', (1, 0), (1, -1), colors.HexColor('#212121')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#81C784')),
            ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.HexColor('#F1F8E9'), colors.white])
        ])
    )
    
    content.append(results_table)
    content.append(Spacer(1, 0.25*inch))
    
    # Add disease information with enhanced formatting
    content.append(Paragraph("Disease Information", custom_styles['SectionHeader']))
    descriptions = prediction["disease_info"]["descriptions"]
    if descriptions:
        for description in descriptions:
            content.append(Paragraph(f"• {description}", custom_styles['BulletPoint']))
    else:
        content.append(Paragraph("No specific disease information available.", custom_styles['Normal']))
    
    # Add treatment recommendations with enhanced formatting
    content.append(Paragraph("Treatment Recommendations", custom_styles['SectionHeader']))
    treatment = prediction["treatment_recommendations"]
    if isinstance(treatment, str):
        paragraphs = treatment.split('\n')
        for para in paragraphs:
            if para.strip():
                content.append(Paragraph(para, custom_styles['Normal']))
    else:
        content.append(Paragraph("No treatment recommendations available.", custom_styles['Normal']))
    
    # Add prevention tips with icons
    content.append(Paragraph("Prevention Tips", custom_styles['SectionHeader']))
    prevention_tips = [
        "- Regularly inspect plants for early signs of disease",
        "- Maintain proper spacing between plants",
        "- Avoid overhead watering to reduce leaf wetness",
        "- Remove and destroy infected plant parts promptly",
        "- Use disease-resistant plant varieties when available"
    ]
    for tip in prevention_tips:
        content.append(Paragraph(tip, custom_styles['BulletPoint']))
    
    # Add reason for disease
    content.append(Paragraph("Reason for Disease", custom_styles['SectionHeader']))
    reason = prediction["reason_for_disease"]
    if isinstance(reason, str):
        paragraphs = reason.split('\n')
        for para in paragraphs:
            if para.strip():
                content.append(Paragraph(para, custom_styles['Normal']))
    else:
        content.append(Paragraph("No information on disease cause available.", custom_styles['Normal']))
    
    # Add warnings with enhanced styling
    if report_data["warnings"]["requires_expert_review"]:
        content.append(Spacer(1, 0.25*inch))
        content.append(Paragraph("- Expert Review Recommended", custom_styles['SectionHeader']))
        
        warning_reasons = []
        if report_data["warnings"]["low_confidence"]:
            warning_reasons.append("❗ Low confidence in prediction")
        if report_data["warnings"]["severe_disease"]:
            warning_reasons.append("- Potentially severe disease detected")
        
        for reason in warning_reasons:
            content.append(Paragraph(reason, custom_styles['BulletPoint']))
    
    # Add footer with enhanced metadata
    content.append(Spacer(1, 0.5*inch))
    footer_text = [
        f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "Plant Disease Classification System v2.0",
        f"Report ID: {report_id}",
        "CONFIDENTIAL - For Internal Use Only"
    ]
    for line in footer_text:
        content.append(Paragraph(line, custom_styles['Footer']))
    
    # Build the PDF with watermark
    doc.build(content, onFirstPage=add_watermark, onLaterPages=add_watermark)
    logger.info(f"Enhanced PDF report generated successfully: {report_path}")
    
    return report_path

@app.on_event("startup")
async def startup_event():
    """Initialize the application and verify setup on startup."""
    try:
        # Initialize the classifier which loads the model
        PlantDiseaseClassifier()
        logger.info("Application initialized successfully")
        
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        raise Exception(f"Failed to initialize application: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    classifier = PlantDiseaseClassifier()
    return {
        "status": "healthy",
        "model_loaded": classifier.model is not None,
        "timestamp": datetime.now().isoformat(),
        "model_info": {
            "path": ModelConfig.MODEL_PATH,
            "num_classes": ModelConfig.NUM_CLASSES
        }
    }

@app.get("/categories")
async def get_categories():
    """Get available plant categories and diseases."""
    classifier = PlantDiseaseClassifier()
    
    return {
        "total_categories": len(classifier.categories),
        "categories": {
            plant: {
                "healthy_samples_available": info["healthy"],
                "diseases": sorted(list(info["diseases"])),
                "total_samples": info["total_samples"]
            }
            for plant, info in classifier.categories.items()
        }
    }

@app.post("/classify")
async def classify_image(
    file: UploadFile = File(...),
    use_gemini: bool = False,
    generate_report: bool = False,
    request: Request = None,
    api_key: str = Depends(get_api_key)
):
    """
    Classify plant disease from uploaded image.
    Returns detailed analysis including confidence scores, disease information, 
    treatment recommendations, and reason for disease.
    Optionally generates a PDF report.
    """
    try:
        # Get host URL for report download link
        base_url = str(request.base_url).rstrip('/')
        
        logger.info(f"Received file: {file.filename}, content type: {file.content_type}")
        
        # Validate file
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an image"
            )
        
        # Read image
        image_bytes = await file.read()
        
        # Validate image data
        logger.info(f"Uploaded file size: {len(image_bytes)} bytes")
        if len(image_bytes) == 0:
            raise HTTPException(
                status_code=400,
                detail="Empty image file"
            )
        
        # Save the image
        save_path = save_upload(file)
        
        # Preprocess image and get metadata
        img_array, metadata = preprocess_image(image_bytes)
        
        # Initialize classifier and make prediction
        classifier = PlantDiseaseClassifier()
        results = classifier.predict(img_array)
        
        # Parse results
        predicted_class = results["top_prediction"]
        confidence = results["confidence"]
        
        plant, condition = predicted_class.split('___')
        plant = plant.replace('_', ' ')
        condition = condition.replace('_', ' ')
        
        # Get disease information
        disease_info = classifier.get_disease_info(condition)
        
        # Load default treatment recommendations and reason for disease
        with open("treatment_recommendations.json", "r") as f:
            treatment_recommendations = json.load(f)
        with open("reason.json", "r") as f:
            reason_for_disease = json.load(f)
        
        # Filter treatment and reason for the predicted class
        treatment = treatment_recommendations.get(predicted_class, "No treatment recommendations available.")
        reason = reason_for_disease.get(predicted_class, "No reason for disease available.")
        
        # If user chooses to use Gemini, override default data
        if use_gemini:
            # Get treatment recommendations from Gemini API
            treatment_prompt = f"Provide detailed treatment recommendations for {condition} in {plant}."
            treatment = get_gemini_response(treatment_prompt)
            
            # Get reason for disease from Gemini API
            reason_prompt = f"Explain the reason for the disease {condition} in {plant}."
            reason = get_gemini_response(reason_prompt)
        
        # Enhanced metadata for the response
        full_metadata = {
            "filename": file.filename,
            "content_type": file.content_type,
            "save_path": str(save_path),
            "timestamp": datetime.now().isoformat(),
            "file_size_bytes": len(image_bytes),
            **metadata  # Include all the metadata from preprocess_image
        }
        
        # Prepare response
        response = {
            "prediction": {
                "plant": plant,
                "condition": condition,
                "confidence": float(round(confidence * 100, 2)),
                "confidence_level": (
                    "High" if confidence > ModelConfig.CONFIDENCE_THRESHOLD
                    else "Medium" if confidence > 0.70
                    else "Low"
                ),
                "disease_info": disease_info,
                "treatment_recommendations": treatment,
                "reason_for_disease": reason,
                "data_source": "Gemini" if use_gemini else "Local"
            },
            "top_3_predictions": results["top_3"],
            "metadata": full_metadata,
            "warnings": {
                "low_confidence": confidence < ModelConfig.CONFIDENCE_THRESHOLD,
                "severe_disease": disease_info["severity"] == "high",
                "requires_expert_review": (
                    confidence < ModelConfig.CONFIDENCE_THRESHOLD or
                    disease_info["severity"] == "high"
                )
            },
            "requires_review": (
                confidence < ModelConfig.CONFIDENCE_THRESHOLD or
                disease_info["severity"] == "high"
            )
        }
        
        # Generate PDF report if requested
        if generate_report:
            try:
                report_path = generate_pdf_report(response, save_path)
                report_filename = os.path.basename(report_path)
                
                # Get PDF file size for report metadata
                report_size_bytes = os.path.getsize(report_path)
                report_size = humanize.naturalsize(report_size_bytes)
                
                # Get current timestamp for report metadata
                generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                response["report"] = {
                    "available": True,
                    "path": str(report_path),
                    "download_url": f"{base_url}/reports/{report_filename}",
                    "filename": report_filename,
                    "size": report_size,
                    "size_bytes": report_size_bytes,
                    "generated_at": generated_at,
                    "page_count": 1  # Default value, could be calculated if needed
                }
            except Exception as e:
                logger.error(f"Error generating PDF report: {str(e)}")
                response["report"] = {
                    "available": False,
                    "error": str(e)
                }
        
        logger.info(
            f"Successfully classified image: {file.filename} "
            f"as {plant} - {condition} "
            f"with {confidence:.2%} confidence"
        )
        
        return JSONResponse(
            content=response,
            headers={"Cache-Control": "no-cache"}
        )
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )

@app.get("/reports/{filename}")
async def get_report(filename: str):
    """Download a generated PDF report."""
    report_path = REPORTS_DIR / filename
    
    if not report_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Report not found"
        )
    
    return FileResponse(
        path=str(report_path),
        media_type="application/pdf",
        filename=filename
    )

# Directory configurations
UPLOAD_DIR = Path("uploads")
REPORTS_DIR = Path("reports")

def cleanup_old_files():
    """Delete files older than 7 days from uploads and reports directories."""
    cutoff_time = datetime.now() - timedelta(days=7)
    
    def delete_old_files_in_dir(directory: Path):
        for file_path in directory.iterdir():
            if file_path.is_file():
                file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_mtime < cutoff_time:
                    try:
                        file_path.unlink()
                        logger.info(f"Deleted old file: {file_path}")
                    except Exception as e:
                        logger.error(f"Failed to delete file {file_path}: {str(e)}")
    
    # Clean up uploads directory
    delete_old_files_in_dir(UPLOAD_DIR)
    
    # Clean up reports directory
    delete_old_files_in_dir(REPORTS_DIR)

# Schedule the cleanup task to run daily
scheduler = BackgroundScheduler()
scheduler.add_job(cleanup_old_files, 'interval', days=1)
scheduler.start()

# Ensure the scheduler shuts down gracefully
@app.on_event("shutdown")
async def shutdown_event():
    scheduler.shutdown()


# Define the router
router = APIRouter()

# Directory configurations
UPLOAD_DIR = Path("uploads")
REPORTS_DIR = Path("reports")

@app.delete("/delete-all-files")
async def delete_all_files():
    """
    Delete all files from the uploads and reports directories immediately.
    """
    try:
        # Function to delete all files in a directory
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

        # Delete files from uploads directory
        deleted_images = delete_files_in_directory(UPLOAD_DIR)

        # Delete files from reports directory
        deleted_reports = delete_files_in_directory(REPORTS_DIR)

        return {
            "message": "All files deleted successfully.",
            "deleted_images": deleted_images,
            "deleted_reports": deleted_reports,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting files: {str(e)}"
        )

# Serve the HTML page at the root URL
@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("static/index.html", "r") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content)

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
async def get_doc():
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