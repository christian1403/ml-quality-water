"""
FastAPI web service for water quality prediction
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import numpy as np
import uvicorn
import sys
import os
import google.generativeai as genai
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.predict import WaterQualityPredictor
from src.utils.analysis_utils import validate_sensor_reading, get_water_quality_guidelines
from config.config import GEMINI_CONFIG, QUALITY_LABELS

app = FastAPI(
    title="Water Quality Prediction API",
    description="ML-powered water quality assessment for consumption safety",
    version="1.0.0"
)

# Global predictor instance
predictor = None

# Initialize Gemini AI
def init_gemini():
    """Initialize Gemini AI with API key"""
    print(GEMINI_CONFIG['api_key'])
    if GEMINI_CONFIG['api_key']:
        try:
            genai.configure(api_key=GEMINI_CONFIG['api_key'])
            return genai.GenerativeModel(GEMINI_CONFIG['model_name'])
        except Exception as e:
            print(f"Failed to initialize Gemini AI: {e}")
            return None
    else:
        print("Warning: Gemini API key not configured. Summary feature will be disabled.")
        return None

# Global Gemini model instance
gemini_model = None

def generate_water_quality_summary(prediction_result: dict) -> Optional[str]:
    """
    Generate human-readable water quality summary using Gemini AI
    
    Args:
        prediction_result: The ML prediction result dictionary
    
    Returns:
        Human-readable summary string or None if Gemini is not available
    """
    global gemini_model
    
    if gemini_model is None:
        return None
    
    try:
        # Extract data from prediction result
        input_data = prediction_result['input']
        prediction = prediction_result['prediction']
        probabilities = prediction_result['probabilities']
        recommendation = prediction_result['recommendation']
        
        # Create a comprehensive prompt
        prompt = f"""
        You are a water quality expert. Based on the following scientific water quality analysis results, provide a clear, human-readable summary for general consumers.

        **Sensor Measurements:**
        - TDS (Total Dissolved Solids): {input_data['tds']} mg/L
        - Turbidity: {input_data['turbidity']} NTU
        - pH Level: {input_data['ph']}

        **AI Model Prediction:**
        - Quality Classification: {prediction['quality_label']}
        - Confidence Level: {prediction['confidence']:.1%}

        **Detailed Probabilities:**
        - Poor Quality: {probabilities['Poor']:.1%}
        - Acceptable Quality: {probabilities['Acceptable']:.1%}
        - Good Quality: {probabilities['Good']:.1%}
        - Excellent Quality: {probabilities['Excellent']:.1%}

        **Technical Recommendation:** {recommendation}

        Please provide a 2-3 sentence summary that:
        1. Explains what these results mean in simple terms
        2. Highlights any concerning parameters
        3. Gives practical advice for the consumer
        4. Uses everyday language (avoid technical jargon)

        Keep it concise, informative, and actionable for someone without technical background. Describe it in Indonesia language
        """
        
        # Generate summary using Gemini
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=GEMINI_CONFIG['temperature'],
                max_output_tokens=GEMINI_CONFIG['max_output_tokens']
            )
        )
        
        return response.text.strip()
        
    except Exception as e:
        print(f"Failed to generate summary with Gemini: {e}")
        return None

class WaterSample(BaseModel):
    """Water sample data model"""
    tds: float = Field(..., description="Total Dissolved Solids (mg/L)", ge=0, le=5000)
    turbidity: float = Field(..., description="Turbidity (NTU)", ge=0, le=100)
    ph: float = Field(..., description="pH level", ge=4.0, le=12.0)

class BatchWaterSamples(BaseModel):
    """Batch water samples data model"""
    samples: List[WaterSample]

@app.on_event("startup")
async def startup_event():
    """Load model and initialize Gemini on startup"""
    global predictor, gemini_model
    
    try:
        # Try to load the predictor - don't fail if models aren't available
        predictor = WaterQualityPredictor()
        print("Water quality prediction model loaded successfully")
    except Exception as e:
        print(f"Warning: Failed to load model: {e}")
        print("API will run in demo mode - some endpoints may not be available")
        predictor = None
    
    # Initialize Gemini AI
    gemini_model = init_gemini()
    if gemini_model:
        print("Gemini AI initialized successfully")
    else:
        print("Gemini AI not available - summary feature disabled")

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "Water Quality Prediction API",
        "version": "1.0.0",
        "status": "ready" if predictor else "demo-mode",
        "endpoints": {
            "/predict": "Single water quality prediction (requires ML model)",
            "/predict/demo": "Demo water quality prediction (rule-based, always available)",
            "/predict/batch": "Batch water quality predictions (requires ML model)",
            "/guidelines": "Water quality guidelines",
            "/health": "API health check"
        },
        "note": "If ML model is not available, use /predict/demo endpoint for testing"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_status = "loaded" if predictor and predictor.model is not None else "not_loaded"
    
    return {
        "status": "healthy",
        "model_status": model_status,
        "api_version": "1.0.0"
    }

@app.post("/predict")
async def predict_water_quality(sample: WaterSample):
    """
    Predict water quality for a single sample
    
    Args:
        sample: Water sample with TDS, turbidity, and pH values
    
    Returns:
        Prediction results with quality class, confidence, recommendations, and AI-generated summary
    """
    if predictor is None or predictor.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Validate sensor readings
        validation = validate_sensor_reading(
            tds=sample.tds, 
            turbidity=sample.turbidity, 
            ph=sample.ph
        )
        
        if not validation['valid']:
            raise HTTPException(status_code=400, detail=validation['errors'])
        
        # Make prediction
        result = predictor.predict(sample.tds, sample.turbidity, sample.ph)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        # Convert NumPy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            """Recursively convert NumPy types to native Python types"""
            if isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif hasattr(obj, 'item'):  # NumPy scalar
                return obj.item()
            elif hasattr(obj, 'tolist'):  # NumPy array
                return obj.tolist()
            else:
                return obj
        
        # Convert all NumPy types in the result
        result = convert_numpy_types(result)
        
        # Add validation warnings if any
        if validation['warnings']:
            result['warnings'] = validation['warnings']
        
        # Generate AI-powered summary using Gemini
        summary = generate_water_quality_summary(result)
        if summary:
            result['summary'] = summary
        else:
            result['summary'] = "AI summary not available. Please check Gemini API configuration."
        
        # Add timestamp
        result['timestamp'] = datetime.now().isoformat()
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch")
async def predict_batch_water_quality(batch: BatchWaterSamples):
    """
    Predict water quality for multiple samples
    
    Args:
        batch: List of water samples
    
    Returns:
        List of prediction results
    """
    if predictor is None or predictor.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        results = []
        
        # Helper function to convert NumPy types
        def convert_numpy_types(obj):
            """Recursively convert NumPy types to native Python types"""
            if isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif hasattr(obj, 'item'):  # NumPy scalar
                return obj.item()
            elif hasattr(obj, 'tolist'):  # NumPy array
                return obj.tolist()
            else:
                return obj
        
        for i, sample in enumerate(batch.samples):
            try:
                # Validate and predict for each sample
                validation = validate_sensor_reading(
                    tds=sample.tds,
                    turbidity=sample.turbidity,
                    ph=sample.ph
                )
                
                if not validation['valid']:
                    results.append({
                        "sample_index": i,
                        "error": validation['errors']
                    })
                    continue
                
                result = predictor.predict(sample.tds, sample.turbidity, sample.ph)
                
                if "error" in result:
                    results.append({
                        "sample_index": i,
                        "error": result['error']
                    })
                else:
                    # Convert NumPy types
                    result = convert_numpy_types(result)
                    result['sample_index'] = i
                    if validation['warnings']:
                        result['warnings'] = validation['warnings']
                    results.append(result)
                    
            except Exception as e:
                results.append({
                    "sample_index": i,
                    "error": f"Processing failed: {str(e)}"
                })
        
        return {"predictions": results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/guidelines")
async def get_guidelines():
    """Get water quality guidelines and standards"""
    guidelines = {
        "standards": {
            "ph": {
                "optimal": "7.0 - 7.5",
                "acceptable": "6.5 - 8.5 (WHO/EPA standard)",
                "note": "Outside range requires treatment"
            },
            "tds": {
                "excellent": "< 300 mg/L",
                "good": "300 - 600 mg/L", 
                "acceptable": "600 - 900 mg/L",
                "poor": "> 900 mg/L"
            },
            "turbidity": {
                "excellent": "< 1 NTU",
                "good": "1 - 4 NTU",
                "acceptable": "4 - 10 NTU", 
                "poor": "> 10 NTU"
            }
        },
        "quality_classes": {
            "3": "Excellent - All parameters in optimal range",
            "2": "Good - Most parameters in good range",
            "1": "Acceptable - Parameters within safe limits",
            "0": "Poor - One or more parameters require attention"
        },
        "sources": ["WHO Guidelines", "EPA Standards", "NSF International"]
    }
    
    return guidelines

@app.post("/analyze")
async def analyze_water_sample(sample: WaterSample):
    """
    Comprehensive water quality analysis
    
    Args:
        sample: Water sample with TDS, turbidity, and pH values
    
    Returns:
        Detailed analysis including ML prediction, parameter analysis, and recommendations
    """
    if predictor is None or predictor.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Helper function to convert NumPy types
        def convert_numpy_types(obj):
            """Recursively convert NumPy types to native Python types"""
            if isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif hasattr(obj, 'item'):  # NumPy scalar
                return obj.item()
            elif hasattr(obj, 'tolist'):  # NumPy array
                return obj.tolist()
            else:
                return obj
        
        # Get ML prediction
        ml_result = predictor.predict(sample.tds, sample.turbidity, sample.ph)
        
        if "error" in ml_result:
            raise HTTPException(status_code=400, detail=ml_result['error'])
        
        # Convert NumPy types in ML result
        ml_result = convert_numpy_types(ml_result)
        
        # Calculate Water Quality Index
        wqi = calculate_water_quality_index(sample.tds, sample.turbidity, sample.ph)
        
        # Parameter analysis
        parameter_analysis = {}
        
        # pH analysis
        if sample.ph < 6.5:
            parameter_analysis['ph'] = f"Acidic ({sample.ph}) - may cause corrosion"
        elif sample.ph > 8.5:
            parameter_analysis['ph'] = f"Alkaline ({sample.ph}) - may cause scaling"
        else:
            parameter_analysis['ph'] = f"Within acceptable range ({sample.ph})"
        
        # TDS analysis
        if sample.tds > 1000:
            parameter_analysis['tds'] = f"High TDS ({sample.tds} mg/L) - may affect taste"
        elif sample.tds < 50:
            parameter_analysis['tds'] = f"Very low TDS ({sample.tds} mg/L) - may lack minerals"
        else:
            parameter_analysis['tds'] = f"Acceptable TDS level ({sample.tds} mg/L)"
        
        # Turbidity analysis
        if sample.turbidity > 4:
            parameter_analysis['turbidity'] = f"High turbidity ({sample.turbidity} NTU) - filtration recommended"
        else:
            parameter_analysis['turbidity'] = f"Acceptable clarity ({sample.turbidity} NTU)"
        
        return {
            "ml_prediction": ml_result,
            "water_quality_index": round(wqi, 2),
            "parameter_analysis": parameter_analysis,
            "overall_assessment": _get_overall_assessment(ml_result['prediction']['quality_class'], wqi)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

def calculate_water_quality_index(tds: float, turbidity: float, ph: float) -> float:
    """
    Calculate Water Quality Index (WQI) based on sensor readings
    
    Args:
        tds: Total Dissolved Solids (mg/L)
        turbidity: Turbidity (NTU)
        ph: pH level
    
    Returns:
        Water Quality Index (0-100 scale)
    """
    # pH sub-index (ideal range: 7.0-7.5)
    if 7.0 <= ph <= 7.5:
        ph_index = 100
    elif 6.5 <= ph <= 8.5:
        ph_index = 80 - abs(ph - 7.25) * 20
    elif 6.0 <= ph <= 9.0:
        ph_index = 60 - abs(ph - 7.25) * 10
    else:
        ph_index = max(0, 40 - abs(ph - 7.25) * 10)
    
    # TDS sub-index
    if tds <= 300:
        tds_index = 100
    elif tds <= 600:
        tds_index = 100 - (tds - 300) / 3
    elif tds <= 900:
        tds_index = 60 - (tds - 600) / 7.5
    else:
        tds_index = max(0, 20 - (tds - 900) / 50)
    
    # Turbidity sub-index
    if turbidity <= 1:
        turbidity_index = 100
    elif turbidity <= 4:
        turbidity_index = 100 - (turbidity - 1) * 10
    elif turbidity <= 10:
        turbidity_index = 60 - (turbidity - 4) * 5
    else:
        turbidity_index = max(0, 30 - (turbidity - 10) * 3)
    
    # Calculate weighted WQI (pH is most critical)
    wqi = (ph_index * 0.4 + tds_index * 0.3 + turbidity_index * 0.3)
    return min(100, max(0, wqi))

def _get_overall_assessment(quality_class, wqi):
    """Generate overall assessment"""
    if quality_class >= 2 and wqi >= 70:
        return "Safe for consumption with excellent/good quality indicators"
    elif quality_class >= 1 and wqi >= 50:
        return "Generally safe for consumption with regular monitoring recommended"
    else:
        return "Treatment or filtration recommended before consumption"

@app.post("/predict/demo")
async def predict_water_quality_demo(sample: WaterSample):
    """
    Demo water quality prediction using rule-based system
    Works without ML model for testing/demo purposes
    
    Args:
        sample: Water sample with TDS, turbidity, and pH values
    
    Returns:
        Demo prediction results with quality assessment
    """
    try:
        # Validate sensor readings
        validation = validate_sensor_reading(
            tds=sample.tds, 
            turbidity=sample.turbidity, 
            ph=sample.ph
        )
        
        if not validation['valid']:
            raise HTTPException(status_code=400, detail=validation['errors'])
        
        # Simple rule-based classification for demo
        score = 0
        
        # pH scoring (0-40 points)
        if 7.0 <= sample.ph <= 7.5:
            score += 40
        elif 6.5 <= sample.ph <= 8.5:
            score += 30
        elif 6.0 <= sample.ph <= 9.0:
            score += 20
        else:
            score += 10
        
        # TDS scoring (0-35 points)
        if sample.tds <= 300:
            score += 35
        elif sample.tds <= 600:
            score += 25
        elif sample.tds <= 900:
            score += 15
        else:
            score += 5
        
        # Turbidity scoring (0-25 points)
        if sample.turbidity <= 1:
            score += 25
        elif sample.turbidity <= 4:
            score += 20
        elif sample.turbidity <= 10:
            score += 10
        else:
            score += 5
        
        # Determine quality class
        if score >= 80:
            quality_class = 3  # Excellent
            confidence = 0.9
        elif score >= 65:
            quality_class = 2  # Good
            confidence = 0.8
        elif score >= 45:
            quality_class = 1  # Acceptable
            confidence = 0.7
        else:
            quality_class = 0  # Poor
            confidence = 0.6
        
        result = {
            "quality_class": quality_class,
            "quality_label": QUALITY_LABELS[quality_class],
            "confidence": confidence,
            "prediction_score": score / 100.0,
            "wqi_score": score,
            "parameters": {
                "tds": sample.tds,
                "turbidity": sample.turbidity,
                "ph": sample.ph
            },
            "assessment": _get_overall_assessment(quality_class, score),
            "recommendations": get_water_quality_guidelines(),
            "mode": "demo",
            "timestamp": datetime.now().isoformat()
        }
        
        # Add validation warnings if any
        if validation['warnings']:
            result['warnings'] = validation['warnings']
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Demo prediction failed: {str(e)}")

if __name__ == "__main__":
    print("Starting Water Quality Prediction API...")
    print("API will be available at: http://localhost:8000")
    print("Interactive documentation: http://localhost:8000/docs")
    
    if os.getenv("ENVIRONMENT") == "production":
        print("Running in production mode")
        uvicorn.run(app)
    else:
        print("Running in development mode")
        uvicorn.run(app, host="0.0.0.0", port=8000)
