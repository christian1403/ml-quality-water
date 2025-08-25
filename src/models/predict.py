"""
Water quality prediction module
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import QUALITY_LABELS
from models.train_model import WaterQualityModel

class WaterQualityPredictor:
    """Production-ready water quality predictor"""
    
    def __init__(self, model_path='models/water_quality_model.h5'):
        self.model_path = model_path
        self.model = None
        self.preprocessor = None
        self.load_model()
    
    def load_model(self):
        """Load trained model and preprocessor"""
        try:
            # Load TensorFlow model
            self.model = tf.keras.models.load_model(self.model_path)
            print(f"Model loaded successfully from {self.model_path}")
            
            # Load preprocessor
            preprocessor_path = self.model_path.replace('.h5', '_preprocessor.pkl')
            from src.data_processing.preprocessor import WaterQualityPreprocessor
            self.preprocessor = WaterQualityPreprocessor.load_preprocessor(preprocessor_path)
            
        except FileNotFoundError:
            print(f"Model files not found. Please train the model first:")
            print("python src/models/train_model.py")
            
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def predict(self, tds, turbidity, ph):
        """
        Predict water quality for given sensor readings
        
        Args:
            tds (float): Total Dissolved Solids (mg/L)
            turbidity (float): Turbidity (NTU)
            ph (float): pH level
        
        Returns:
            dict: Prediction results with quality class, label, and confidence
        """
        if self.model is None or self.preprocessor is None:
            return {"error": "Model not loaded properly"}
        
        try:
            # Validate input ranges
            if not (4.0 <= ph <= 12.0):
                return {"error": f"pH value {ph} is outside valid range (4.0-12.0)"}
            
            if not (0 <= tds <= 5000):
                return {"error": f"TDS value {tds} is outside valid range (0-5000 mg/L)"}
                
            if not (0 <= turbidity <= 100):
                return {"error": f"Turbidity value {turbidity} is outside valid range (0-100 NTU)"}
            
            # Preprocess input
            sample_scaled = self.preprocessor.preprocess_single_sample(tds, turbidity, ph)
            
            # Make prediction
            pred_proba = self.model.predict(sample_scaled, verbose=0)
            pred_class = np.argmax(pred_proba, axis=1)[0]
            confidence = pred_proba[0][pred_class]
            
            # Prepare results
            result = {
                'input': {
                    'tds': tds,
                    'turbidity': turbidity,
                    'ph': ph
                },
                'prediction': {
                    'quality_class': int(pred_class),
                    'quality_label': QUALITY_LABELS[pred_class],
                    'confidence': float(confidence)
                },
                'probabilities': {
                    QUALITY_LABELS[i]: float(pred_proba[0][i]) 
                    for i in range(4)
                },
                'recommendation': self._get_recommendation(pred_class, confidence)
            }
            
            return result
            
        except Exception as e:
            return {"error": f"Prediction failed: {e}"}
    
    def _get_recommendation(self, quality_class, confidence):
        """Generate recommendation based on prediction"""
        if confidence < 0.7:
            return "Low confidence prediction. Consider additional testing."
        
        if quality_class == 3:
            return "Excellent water quality. Safe for consumption."
        elif quality_class == 2:
            return "Good water quality. Generally safe for consumption."
        elif quality_class == 1:
            return "Acceptable water quality. Monitor regularly and consider treatment."
        else:
            return "Poor water quality. Treatment required before consumption."
    
    def predict_batch(self, data):
        """
        Predict water quality for multiple samples
        
        Args:
            data (pd.DataFrame or list): Data with columns ['tds', 'turbidity', 'ph']
        
        Returns:
            list: List of prediction results
        """
        if isinstance(data, list):
            results = []
            for sample in data:
                if len(sample) != 3:
                    results.append({"error": "Each sample must have 3 values: [tds, turbidity, ph]"})
                    continue
                
                result = self.predict(sample[0], sample[1], sample[2])
                results.append(result)
            
            return results
        
        elif isinstance(data, pd.DataFrame):
            results = []
            for _, row in data.iterrows():
                result = self.predict(row['tds'], row['turbidity'], row['ph'])
                results.append(result)
            
            return results
        
        else:
            return [{"error": "Data must be a list or pandas DataFrame"}]
    
    def generate_report(self, tds, turbidity, ph):
        """Generate detailed water quality report"""
        prediction = self.predict(tds, turbidity, ph)
        
        if "error" in prediction:
            return prediction
        
        report = f"""
=== WATER QUALITY ANALYSIS REPORT ===

Sensor Readings:
- TDS (Total Dissolved Solids): {tds} mg/L
- Turbidity: {turbidity} NTU
- pH Level: {ph}

Prediction Results:
- Quality Classification: {prediction['prediction']['quality_label']}
- Confidence: {prediction['prediction']['confidence']:.2%}

Detailed Probabilities:
"""
        
        for label, prob in prediction['probabilities'].items():
            report += f"- {label}: {prob:.2%}\n"
        
        report += f"\nRecommendation: {prediction['recommendation']}\n"
        
        # Add parameter analysis
        report += "\nParameter Analysis:\n"
        report += self._analyze_parameters(tds, turbidity, ph)
        
        return report
    
    def _analyze_parameters(self, tds, turbidity, ph):
        """Analyze individual parameters against standards"""
        analysis = ""
        
        # pH Analysis
        if 7.0 <= ph <= 7.5:
            analysis += "- pH: Optimal range (7.0-7.5)\n"
        elif 6.5 <= ph <= 8.5:
            analysis += "- pH: Acceptable range (6.5-8.5)\n"
        else:
            analysis += "- pH: Outside acceptable range (6.5-8.5) - requires attention\n"
        
        # TDS Analysis
        if tds <= 300:
            analysis += "- TDS: Excellent (<300 mg/L)\n"
        elif tds <= 600:
            analysis += "- TDS: Good (300-600 mg/L)\n"
        elif tds <= 900:
            analysis += "- TDS: Acceptable (600-900 mg/L)\n"
        else:
            analysis += "- TDS: Poor (>900 mg/L) - treatment recommended\n"
        
        # Turbidity Analysis
        if turbidity <= 1:
            analysis += "- Turbidity: Excellent (<1 NTU)\n"
        elif turbidity <= 4:
            analysis += "- Turbidity: Good (1-4 NTU)\n"
        elif turbidity <= 10:
            analysis += "- Turbidity: Acceptable (4-10 NTU)\n"
        else:
            analysis += "- Turbidity: Poor (>10 NTU) - filtration needed\n"
        
        return analysis

def main():
    """Interactive prediction interface"""
    predictor = WaterQualityPredictor()
    
    if predictor.model is None:
        return
    
    print("=== Water Quality Predictor ===")
    print("Enter sensor readings to predict water quality")
    print("(Type 'quit' to exit)")
    
    while True:
        try:
            print("\n" + "="*50)
            
            # Get user input
            user_input = input("Enter command (predict/batch/quit): ").strip().lower()
            
            if user_input == 'quit':
                break
            
            elif user_input == 'predict':
                # Single prediction
                tds = float(input("Enter TDS (mg/L): "))
                turbidity = float(input("Enter Turbidity (NTU): "))
                ph = float(input("Enter pH: "))
                
                report = predictor.generate_report(tds, turbidity, ph)
                print(report)
            
            elif user_input == 'batch':
                # Batch prediction example
                print("Running batch prediction on sample data...")
                
                sample_data = [
                    [250, 0.8, 7.2],  # Should be excellent
                    [800, 5.0, 6.8],  # Should be acceptable
                    [1500, 15.0, 5.5],  # Should be poor
                    [400, 2.0, 7.0]   # Should be good
                ]
                
                results = predictor.predict_batch(sample_data)
                
                for i, result in enumerate(results):
                    if "error" not in result:
                        print(f"\nSample {i+1}: {sample_data[i]}")
                        print(f"Quality: {result['prediction']['quality_label']}")
                        print(f"Confidence: {result['prediction']['confidence']:.2%}")
                    else:
                        print(f"Sample {i+1}: {result['error']}")
            
            else:
                print("Invalid command. Use 'predict', 'batch', or 'quit'")
                
        except ValueError:
            print("Invalid input. Please enter numeric values.")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
