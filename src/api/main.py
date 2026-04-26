"""
FastAPI backend for Diabetic Retinopathy Screening System.
Provides REST API for inference and report generation.
"""

import os
import io
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yaml

from src.inference.inference import DRInference
from src.reports.generator import PDFReportGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load configuration
with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# Initialize FastAPI app
app = FastAPI(
    title="Diabetic Retinopathy Screening System",
    description="AI-powered DR detection from retinal fundus images",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
inference_engine: Optional[DRInference] = None
report_generator: Optional[PDFReportGenerator] = None
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)


# Pydantic models
class PredictionResponse(BaseModel):
    success: bool
    predicted_class: int
    label: str
    confidence: float
    probabilities: dict
    referral_needed: bool
    quality_check: Optional[dict] = None
    message: str


class PatientInfo(BaseModel):
    patient_id: str
    patient_name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    examination_date: Optional[str] = None
    notes: Optional[str] = None


class ReportResponse(BaseModel):
    success: bool
    report_path: Optional[str] = None
    message: str


@app.on_event("startup")
async def startup_event():
    """Initialize models and services on startup."""
    global inference_engine, report_generator
    
    logger.info("Initializing DR Screening System...")
    
    # Load model
    model_path = config.get('MODEL_PATH', 'models/checkpoints/best_model.ckpt')
    if os.path.exists(model_path):
        inference_engine = DRInference(
            model_path=model_path,
            config=config,
            use_onnx=False
        )
        logger.info(f"Model loaded from {model_path}")
    else:
        logger.warning(f"Model not found at {model_path}. Predictions will not work.")
    
    # Initialize report generator
    report_generator = PDFReportGenerator(config)
    logger.info("Report generator initialized")
    
    logger.info("System ready")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Diabetic Retinopathy Screening System",
        "version": "1.0.0",
        "model_loaded": inference_engine is not None
    }


@app.get("/health")
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": inference_engine is not None,
        "config": {
            "backbone": config['MODEL']['BACKBONE'],
            "image_size": config['MODEL']['IMAGE_SIZE'],
            "num_classes": config['MODEL']['NUM_CLASSES']
        }
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict DR grade from uploaded retinal image.
    
    - **file**: Retinal fundus image (JPG, PNG, BMP)
    """
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in config['API']['ALLOWED_EXTENSIONS']:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {config['API']['ALLOWED_EXTENSIONS']}"
        )
    
    # Save uploaded file
    file_id = str(uuid.uuid4())
    temp_path = os.path.join(TEMP_DIR, f"{file_id}{file_extension}")
    
    try:
        contents = await file.read()
        
        # Check file size
        if len(contents) > config['API']['MAX_IMAGE_SIZE']:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Max size: {config['API']['MAX_IMAGE_SIZE'] / 1e6:.0f}MB"
            )
        
        with open(temp_path, "wb") as f:
            f.write(contents)
        
        # Run prediction
        result = inference_engine.predict(temp_path, return_heatmap=True)
        
        # Save heatmap if generated
        heatmap_path = None
        overlay_path = None
        if 'heatmap' in result:
            heatmap_path = os.path.join(TEMP_DIR, f"{file_id}_heatmap.png")
            overlay_path = os.path.join(TEMP_DIR, f"{file_id}_overlay.png")
            
            inference_engine.save_heatmap(result['heatmap'], heatmap_path)
            inference_engine.save_overlay(result['overlay'], overlay_path)
        
        # Prepare response
        probabilities_dict = {
            f"Class {i} ({inference_engine.DR_LABELS[i]})": result['probabilities'][i]
            for i in range(5)
        }
        
        response = PredictionResponse(
            success=True,
            predicted_class=result['predicted_class'],
            label=result['label'],
            confidence=result['confidence'],
            probabilities=probabilities_dict,
            referral_needed=result['referral_needed'],
            message="Urgent referral recommended" if result['referral_needed'] else "No immediate referral needed"
        )
        
        # Store result for report generation
        result['heatmap_path'] = heatmap_path
        result['overlay_path'] = overlay_path
        result['file_id'] = file_id
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/report", response_model=ReportResponse)
async def generate_report(
    patient_info: PatientInfo,
    background_tasks: BackgroundTasks
):
    """
    Generate PDF report for a prediction.
    
    - **patient_info**: Patient information
    """
    if report_generator is None:
        raise HTTPException(status_code=503, detail="Report generator not initialized")
    
    try:
        # Find latest prediction result
        # In production, this should be stored in a database
        latest_file_id = patient_info.patient_id
        
        # Generate report
        report_path = report_generator.generate_report(
            patient_info=patient_info.dict(),
            prediction_result={},  # Should be retrieved from database
            output_dir="reports"
        )
        
        return ReportResponse(
            success=True,
            report_path=report_path,
            message="Report generated successfully"
        )
    
    except Exception as e:
        logger.error(f"Report generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/report/{report_id}")
async def download_report(report_id: str):
    """Download a generated report."""
    report_path = os.path.join("reports", f"{report_id}.pdf")
    
    if not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="Report not found")
    
    return FileResponse(
        report_path,
        media_type='application/pdf',
        filename=f"DR_Report_{report_id}.pdf"
    )


@app.get("/classes")
async def get_classes():
    """Get DR class information."""
    return {
        "classes": {
            "0": {"name": "No DR", "description": "No diabetic retinopathy detected", "referral": False},
            "1": {"name": "Mild DR", "description": "Mild non-proliferative DR", "referral": False},
            "2": {"name": "Moderate DR", "description": "Moderate non-proliferative DR", "referral": False},
            "3": {"name": "Severe DR", "description": "Severe non-proliferative DR", "referral": True},
            "4": {"name": "Proliferative DR", "description": "Proliferative DR - Vision threatening", "referral": True}
        }
    }


@app.post("/batch-predict")
async def batch_predict(files: list[UploadFile] = File(...)):
    """
    Predict DR grade for multiple images.
    
    - **files**: List of retinal fundus images
    """
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    
    for file in files:
        try:
            # Similar to single prediction
            file_extension = Path(file.filename).suffix.lower()
            file_id = str(uuid.uuid4())
            temp_path = os.path.join(TEMP_DIR, f"{file_id}{file_extension}")
            
            contents = await file.read()
            with open(temp_path, "wb") as f:
                f.write(contents)
            
            result = inference_engine.predict(temp_path)
            result['filename'] = file.filename
            results.append(result)
            
            os.remove(temp_path)
            
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {e}")
            results.append({
                'filename': file.filename,
                'error': str(e)
            })
    
    return {
        "success": True,
        "count": len(results),
        "results": results
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api.main:app",
        host=config['API']['HOST'],
        port=config['API']['PORT'],
        reload=True
    )
