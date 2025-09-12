"""
TensorFlow Lite Water Quality Predictor
Lightweight prediction module using TensorFlow Lite runtime for serverless deployment
"""

import numpy as np
import sys
import os

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from config.config import QUALITY_LABELS

# Try to import TensorFlow Lite runtime (preferred for deployment)
try:
    import tflite_runtime.interpreter as tflite
    TFLITE_RUNTIME_AVAILABLE = True
    print("🚀 Using TensorFlow Lite runtime")
except ImportError:
    # Fallback to full TensorFlow if TFLite runtime not available
    try:
        import tensorflow as tf
        tflite = tf.lite
        TFLITE_RUNTIME_AVAILABLE = False
        print("⚠️  TensorFlow Lite runtime not found, using full TensorFlow")
    except ImportError:
        print("❌ Neither TensorFlow Lite runtime nor TensorFlow found!")
        raise ImportError("Please install tflite-runtime or tensorflow")

class TFLiteWaterQualityPredictor:
    """Lightweight water quality predictor using TensorFlow Lite"""
    
    def __init__(self, model_path='models/water_quality_model_quantized.tflite'):
        self.model_path = model_path
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.preprocessor = None
        self.load_model()
        self.load_preprocessor()
    
    def load_model(self):
        """Load TensorFlow Lite model"""
        try:
            if not os.path.exists(self.model_path):
                print(f"❌ TFLite model not found: {self.model_path}")
                print("📝 Please run the conversion script first:")
                print("   python src/models/convert_to_tflite.py")
                return
            
            # Load TFLite model
            if TFLITE_RUNTIME_AVAILABLE:
                self.interpreter = tflite.Interpreter(model_path=self.model_path)
            else:
                self.interpreter = tflite.Interpreter(model_path=self.model_path)
            
            self.interpreter.allocate_tensors()
            
            # Get input and output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            print(f"✅ TFLite model loaded: {self.model_path}")
            print(f"   - Input shape: {self.input_details[0]['shape']}")
            print(f"   - Output shape: {self.output_details[0]['shape']}")
            print(f"   - Model size: {os.path.getsize(self.model_path) / 1024:.1f} KB")
            
        except Exception as e:
            print(f"❌ Error loading TFLite model: {e}")
            self.interpreter = None
    
    def load_preprocessor(self):
        """Load preprocessor for feature engineering"""
        try:
            preprocessor_path = self.model_path.replace('.tflite', '_preprocessor.pkl').replace('_quantized', '')
            
            if os.path.exists(preprocessor_path):
                from src.data_processing.preprocessor import WaterQualityPreprocessor
                self.preprocessor = WaterQualityPreprocessor.load_preprocessor(preprocessor_path)
                print(f"✅ Preprocessor loaded: {preprocessor_path}")
            else:
                print(f"⚠️  Preprocessor not found: {preprocessor_path}")
                print("   Using basic preprocessing")
                
        except Exception as e:
            print(f"⚠️  Error loading preprocessor: {e}")
            self.preprocessor = None
    
    def preprocess_input(self, tds, turbidity, ph):
        """
        Preprocess input data for TFLite model
        
        Args:
            tds (float): Total Dissolved Solids (mg/L)
            turbidity (float): Turbidity (NTU)
            ph (float): pH level
        
        Returns:
            np.ndarray: Preprocessed input ready for TFLite model
        """
        try:
            if self.preprocessor is not None:
                # Use full feature engineering if preprocessor is available
                processed = self.preprocessor.preprocess_single_sample(tds, turbidity, ph)
            else:
                # Basic preprocessing if no preprocessor available
                # Simple normalization (basic fallback)
                processed = np.array([[
                    (tds - 500) / 1000,      # Rough normalization for TDS
                    (turbidity - 5) / 10,    # Rough normalization for turbidity
                    (ph - 7) / 2             # Rough normalization for pH
                ]], dtype=np.float32)
            
            # Ensure correct shape and dtype for TFLite
            input_shape = self.input_details[0]['shape']
            input_dtype = self.input_details[0]['dtype']
            
            # Reshape if needed
            if processed.shape != tuple(input_shape):
                processed = processed.reshape(input_shape)
            
            # Convert dtype if needed
            if processed.dtype != input_dtype:
                processed = processed.astype(input_dtype)
            
            return processed
            
        except Exception as e:
            print(f"❌ Error in preprocessing: {e}")
            # Return basic normalized input as fallback
            return np.array([[
                (tds - 500) / 1000,
                (turbidity - 5) / 10,
                (ph - 7) / 2
            ]], dtype=np.float32)
    
    def predict(self, tds, turbidity, ph):
        """
        Predict water quality using TensorFlow Lite model
        
        Args:
            tds (float): Total Dissolved Solids (mg/L)
            turbidity (float): Turbidity (NTU)
            ph (float): pH level
        
        Returns:
            dict: Prediction results with quality class, confidence, and probabilities
        """
        if self.interpreter is None:
            return {"error": "TFLite model not loaded properly"}
        
        try:
            # Validate input ranges
            if not (4.0 <= ph <= 12.0):
                return {"error": f"pH value {ph} is outside valid range (4.0-12.0)"}
            
            if not (0 <= tds <= 5000):
                return {"error": f"TDS value {tds} is outside valid range (0-5000 mg/L)"}
                
            if not (0 <= turbidity <= 100):
                return {"error": f"Turbidity value {turbidity} is outside valid range (0-100 NTU)"}
            
            # Preprocess input
            input_data = self.preprocess_input(tds, turbidity, ph)
            
            # Set input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            
            # Run inference
            self.interpreter.invoke()
            
            # Get output
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # Process output probabilities
            probabilities = output_data[0]  # Remove batch dimension
            
            # Ensure probabilities sum to 1 (apply softmax if needed)
            if abs(np.sum(probabilities) - 1.0) > 0.01:
                # Apply softmax normalization
                exp_probs = np.exp(probabilities - np.max(probabilities))
                probabilities = exp_probs / np.sum(exp_probs)
            
            # Get predicted class and confidence
            pred_class = int(np.argmax(probabilities))
            confidence = float(probabilities[pred_class])
            
            # Apply confidence calibration based on parameter consistency
            calibrated_confidence = self._calibrate_confidence(
                tds, turbidity, ph, pred_class, confidence
            )
            
            # Prepare results
            result = {
                'input': {
                    'tds': float(tds),
                    'turbidity': float(turbidity),
                    'ph': float(ph)
                },
                'prediction': {
                    'quality_class': pred_class,
                    'quality_label': QUALITY_LABELS[pred_class],
                    'confidence': calibrated_confidence
                },
                'probabilities': {
                    QUALITY_LABELS[i]: float(probabilities[i]) 
                    for i in range(len(QUALITY_LABELS))
                },
                'recommendation': self._get_recommendation(pred_class, calibrated_confidence),
                'model_type': 'tflite',
                'model_size_kb': round(os.path.getsize(self.model_path) / 1024, 1)
            }
            
            return result
            
        except Exception as e:
            return {"error": f"TFLite prediction failed: {e}"}
    
    def _calibrate_confidence(self, tds, turbidity, ph, pred_class, base_confidence):
        """
        Calibrate confidence based on parameter consistency
        
        Args:
            tds, turbidity, ph: Input parameters
            pred_class: Predicted class
            base_confidence: Base model confidence
        
        Returns:
            float: Calibrated confidence
        """
        try:
            # Calculate parameter-based quality scores
            ph_score = self._get_parameter_score('ph', ph)
            tds_score = self._get_parameter_score('tds', tds)
            turbidity_score = self._get_parameter_score('turbidity', turbidity)
            
            # Average parameter quality
            avg_quality = (ph_score + tds_score + turbidity_score) / 3.0
            
            # Calculate consistency between parameters and prediction
            consistency = 1.0 - abs(avg_quality - pred_class) / 3.0
            consistency = max(0.0, min(1.0, consistency))
            
            # Adjust confidence based on consistency
            if consistency > 0.8:
                # High consistency - boost confidence
                calibrated = min(0.95, base_confidence + 0.1)
            elif consistency > 0.6:
                # Medium consistency - slight boost
                calibrated = min(0.9, base_confidence + 0.05)
            elif consistency < 0.3:
                # Low consistency - reduce confidence
                calibrated = max(0.4, base_confidence - 0.15)
            else:
                # Normal consistency
                calibrated = base_confidence
            
            return float(calibrated)
            
        except Exception:
            # Return base confidence if calibration fails
            return float(base_confidence)
    
    def _get_parameter_score(self, parameter, value):
        """Get quality score (0-3) for a parameter"""
        if parameter == 'ph':
            if 7.0 <= value <= 7.5:
                return 3  # Excellent
            elif 6.5 <= value <= 8.5:
                return 2  # Good
            elif 6.0 <= value <= 9.0:
                return 1  # Acceptable
            else:
                return 0  # Poor
                
        elif parameter == 'tds':
            if value <= 300:
                return 3  # Excellent
            elif value <= 600:
                return 2  # Good
            elif value <= 900:
                return 1  # Acceptable
            else:
                return 0  # Poor
                
        elif parameter == 'turbidity':
            if value <= 1:
                return 3  # Excellent
            elif value <= 4:
                return 2  # Good
            elif value <= 10:
                return 1  # Acceptable
            else:
                return 0  # Poor
        
        return 1  # Default
    
    def _get_recommendation(self, quality_class, confidence):
        """Generate recommendation based on prediction"""
        if confidence < 0.5:
            return "Low confidence prediction. Consider additional testing."
        
        recommendations = {
            3: "Excellent water quality. Safe for consumption.",
            2: "Good water quality. Generally safe for consumption.",
            1: "Acceptable water quality. Monitor regularly and consider treatment.",
            0: "Poor water quality. Treatment required before consumption."
        }
        
        return recommendations.get(quality_class, "Unknown quality level.")
    
    def get_model_info(self):
        """Get information about the loaded TFLite model"""
        if self.interpreter is None:
            return {"error": "Model not loaded"}
        
        return {
            "model_path": self.model_path,
            "model_size_kb": round(os.path.getsize(self.model_path) / 1024, 1),
            "input_shape": self.input_details[0]['shape'].tolist(),
            "output_shape": self.output_details[0]['shape'].tolist(),
            "input_dtype": str(self.input_details[0]['dtype']),
            "output_dtype": str(self.output_details[0]['dtype']),
            "runtime": "tflite_runtime" if TFLITE_RUNTIME_AVAILABLE else "tensorflow",
            "preprocessor_available": self.preprocessor is not None
        }

def main():
    """Test the TFLite predictor"""
    print("🧪 Testing TensorFlow Lite Water Quality Predictor...")
    print("=" * 60)
    
    predictor = TFLiteWaterQualityPredictor()
    
    if predictor.interpreter is None:
        print("❌ Cannot test - model not loaded")
        return
    
    # Print model info
    model_info = predictor.get_model_info()
    print("📊 Model Information:")
    for key, value in model_info.items():
        print(f"   - {key}: {value}")
    print()
    
    # Test with sample data
    test_samples = [
        (250, 1.0, 7.2, "Excellent quality"),
        (450, 3.0, 7.0, "Good quality"),
        (750, 8.0, 6.8, "Acceptable quality"), 
        (1200, 15.0, 5.5, "Poor quality"),
        (5000, 100, 4.0, "Extreme values")
    ]
    
    print("🧪 Testing predictions:")
    for tds, turbidity, ph, description in test_samples:
        result = predictor.predict(tds, turbidity, ph)
        
        if "error" in result:
            print(f"❌ {description}: {result['error']}")
        else:
            print(f"✅ {description}:")
            print(f"   - Input: TDS={tds}, Turbidity={turbidity}, pH={ph}")
            print(f"   - Quality: {result['prediction']['quality_label']}")
            print(f"   - Confidence: {result['prediction']['confidence']:.2%}")
        print()
    
    print("🎉 TensorFlow Lite predictor test completed!")

if __name__ == "__main__":
    main()
