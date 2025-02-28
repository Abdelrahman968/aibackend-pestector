# Documentation - Plant Disease Classifier

## Introduction

The `main.py` file is the core of the Plant Disease Classifier application. It is built using FastAPI, a modern, fast (high-performance) web framework for building APIs with Python 3.7+ based on standard Python type hints.

This application uses a pre-trained deep learning model (VGG16) to classify plant diseases from uploaded images. It provides a RESTful API for image classification, generates detailed reports, and offers additional features like treatment recommendations and disease information.

## Key Features

*   Image upload and disease classification
*   Detailed disease information and treatment recommendations
*   PDF report generation
*   API key authentication
*   Automatic cleanup of old files
*   Integration with Gemini API for advanced recommendations

## Code Structure

The code is organized into several sections, each handling a specific part of the application:

*   **Imports:** All necessary libraries and modules are imported at the beginning.
*   **Logging Configuration:** Logging is configured to capture both file and console outputs.
*   **FastAPI App Initialization:** The FastAPI app is initialized with CORS middleware and static file serving.
*   **API Key Authentication:** API key authentication is implemented to secure the endpoints.
*   **Model Configuration:** The model path, image size, and other configurations are defined.
*   **PlantDiseaseClassifier Class:** This class handles the loading of the model, prediction, and disease information retrieval.
*   **Endpoints:** Various endpoints are defined for image classification, report generation, and file management.
*   **Background Tasks:** A background scheduler is used to clean up old files periodically.

## Detailed Explanation

### Logging Configuration

Logging is set up to capture both file and console outputs. This helps in debugging and monitoring the application.

```

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
                
```

### FastAPI App Initialization

The FastAPI app is initialized with a title, description, and version. CORS middleware is added to allow cross-origin requests.

```

app = FastAPI(
    title="Plant Disease Classifier",
    description="AI-powered plant disease classification system",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
                
```

### API Key Authentication

API key authentication is implemented using the `APIKeyHeader` dependency. Only requests with valid API keys are processed.

```

API_KEY_NAME = "X-API-KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

VALID_API_KEYS = {"1122333", "445566"}

async def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API Key"
        )
    return api_key
                
```

### Model Configuration

The model configuration includes the path to the pre-trained model, image size, number of classes, and confidence threshold.

```

class ModelConfig:
    MODEL_PATH = "model/vgg_model.h5"  
    IMAGE_SIZE = (224, 224)
    NUM_CLASSES = 38
    CONFIDENCE_THRESHOLD = 0.85
                
```

### PlantDiseaseClassifier Class

This class is responsible for loading the model, making predictions, and organizing disease information. It uses a singleton pattern to ensure the model is loaded only once.

```

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
            ...
        ]
        self.categories = self.organize_categories()
        self.disease_patterns = {
            "blight": {
                "keywords": ["blight", "brown", "lesions"],
                "description": "Causes brown lesions and tissue death",
                "severity": "high"
            },
            ...
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
                
```

### Endpoints

The application provides several endpoints for different functionalities:

*   `/health`: Health check endpoint.
*   `/categories`: Get available plant categories and diseases.
*   `/classify`: Classify plant disease from uploaded image.
*   `/reports/{filename}`: Download a generated PDF report.
*   `/delete-all-files`: Delete all files from the uploads and reports directories.

### Background Tasks

A background scheduler is used to clean up old files periodically. This task runs daily and deletes files older than 7 days.

```

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
                
```

## Conclusion

The `main.py` file is a comprehensive implementation of a plant disease classification system. It leverages modern web technologies, machine learning, and background tasks to provide a robust and scalable solution for plant disease detection and management.
