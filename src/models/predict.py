"""
Water quality prediction module with advanced confidence calibration
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
    """Production-ready water quality predictor with confidence calibration"""
    
    def __init__(self, model_path='models/water_quality_model.h5'):
        self.model_path = model_path
        self.model = None
        self.preprocessor = None
        self.calibrator = None
        self.load_model()
        self.load_calibrator()
    
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
    
    def load_calibrator(self):
        """Load confidence calibrator if available"""
        try:
            from src.models.calibration import ConfidenceCalibrator
            calibrator_path = 'models/confidence_calibrator.pkl'
            
            if os.path.exists(calibrator_path):
                self.calibrator = ConfidenceCalibrator()
                if self.calibrator.load(calibrator_path):
                    print("🎯 Confidence calibrator loaded successfully")
                else:
                    self.calibrator = None
            else:
                print("📝 No calibrator found. Will use standard confidence.")
                self.calibrator = None
                
        except Exception as e:
            print(f"⚠️  Error loading calibrator: {e}")
            self.calibrator = None
    
    def predict(self, tds, turbidity, ph):
        """
        Predict water quality for given sensor readings with enhanced confidence
        
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
            
            # Preprocess input with feature engineering support
            sample_scaled = self.preprocessor.preprocess_single_sample(tds, turbidity, ph)
            
            # Make prediction - get raw logits for calibration
            pred_proba = self.model.predict(sample_scaled, verbose=0)
            
            # Apply confidence calibration if available
            if self.calibrator is not None:
                # Convert probabilities to logits for calibration
                epsilon = 1e-15
                pred_proba_clipped = np.clip(pred_proba, epsilon, 1 - epsilon)
                logits = np.log(pred_proba_clipped)
                
                # Apply calibration
                calibrated_proba = self.calibrator.calibrate_probabilities(logits)
                pred_proba = calibrated_proba
            
            pred_class = np.argmax(pred_proba, axis=1)[0]
            base_confidence = pred_proba[0][pred_class]
            
            # Apply confidence enhancement based on water quality standards
            enhanced_confidence, adjusted_class = self._enhance_confidence_with_standards(
                tds, turbidity, ph, pred_class, base_confidence, pred_proba[0]
            )
            
            # Apply AGGRESSIVE confidence enhancement
            from src.models.confidence_booster import confidence_booster
            print(f"\n🚀 Applying Enhanced Confidence Boosting...")
            print(f"📊 Base confidence: {enhanced_confidence:.2%}")
            
            final_confidence = confidence_booster.enhance_confidence(
                tds, turbidity, ph, pred_class, enhanced_confidence, pred_proba[0]
            )
            
            # Use enhanced predictions
            final_class = adjusted_class if adjusted_class is not None else pred_class
            
            # Prepare results
            result = {
                'input': {
                    'tds': tds,
                    'turbidity': turbidity,
                    'ph': ph
                },
                'prediction': {
                    'quality_class': int(final_class),
                    'quality_label': QUALITY_LABELS[final_class],
                    'confidence': float(final_confidence)
                },
                'probabilities': {
                    QUALITY_LABELS[i]: float(pred_proba[0][i]) 
                    for i in range(4)
                },
                'recommendation': self._get_recommendation(final_class, final_confidence),
                'enhancement_applied': final_confidence > base_confidence
            }
            
            return result
            
        except Exception as e:
            return {"error": f"Prediction failed: {e}"}
    
    def _enhance_confidence_with_standards(self, tds, turbidity, ph, pred_class, base_confidence, probabilities):
        """
        Enhance prediction confidence using WHO/EPA water quality standards
        
        Args:
            tds: Total Dissolved Solids (mg/L)
            turbidity: Turbidity (NTU)
            ph: pH level
            pred_class: Original predicted class
            base_confidence: Original confidence
            probabilities: All class probabilities
        
        Returns:
            Tuple: (enhanced_confidence, adjusted_class)
        """
        # Import water quality standards
        from config.config import WATER_STANDARDS
        
        # Calculate individual parameter quality scores
        ph_score = self._get_parameter_quality_score('ph', ph)
        tds_score = self._get_parameter_quality_score('tds', tds)
        turbidity_score = self._get_parameter_quality_score('turbidity', turbidity)
        
        # Calculate composite water quality index (0-3 scale)
        wqi = (ph_score + tds_score + turbidity_score) / 3.0
        
        # Determine rule-based classification
        rule_based_class = int(round(wqi))
        rule_based_class = max(0, min(3, rule_based_class))  # Ensure valid range
        
        # Calculate confidence enhancement based on parameter alignment
        parameter_consistency = self._calculate_parameter_consistency(tds, turbidity, ph, pred_class)
        
        # Enhanced confidence calculation
        if abs(rule_based_class - pred_class) <= 1:  # Rule-based supports ML prediction
            # Boost confidence based on parameter consistency
            confidence_boost = 0.15 + (parameter_consistency * 0.25)
            enhanced_confidence = min(0.95, base_confidence + confidence_boost)
            
            # If rule-based and ML agree exactly, boost even more
            if rule_based_class == pred_class:
                enhanced_confidence = min(0.95, enhanced_confidence + 0.1)
                
            return enhanced_confidence, pred_class
            
        else:  # Significant disagreement between rule-based and ML
            # Use weighted average of ML and rule-based predictions
            ml_weight = base_confidence
            rule_weight = parameter_consistency
            total_weight = ml_weight + rule_weight
            
            if total_weight > 0:
                # Weighted decision
                if rule_weight > ml_weight * 1.5:  # Rule-based is much stronger
                    return min(0.85, 0.6 + parameter_consistency * 0.25), rule_based_class
                else:  # Keep ML prediction but with adjusted confidence
                    return min(0.75, base_confidence + 0.1), pred_class
            else:
                return base_confidence, pred_class
    
    def _get_parameter_quality_score(self, parameter, value):
        """Get quality score (0-3) for a parameter based on WHO/EPA standards"""
        from config.config import WATER_STANDARDS
        
        standards = WATER_STANDARDS[parameter]
        
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
        
        return 1  # Default acceptable
    
    def _calculate_parameter_consistency(self, tds, turbidity, ph, predicted_class):
        """
        Calculate how consistent the individual parameters are with the predicted class
        Returns a consistency score between 0 and 1
        """
        ph_score = self._get_parameter_quality_score('ph', ph)
        tds_score = self._get_parameter_quality_score('tds', turbidity)
        turbidity_score = self._get_parameter_quality_score('turbidity', turbidity)
        
        # Average parameter score
        avg_score = (ph_score + tds_score + turbidity_score) / 3.0
        
        # Calculate consistency (how close avg_score is to predicted_class)
        consistency = 1.0 - abs(avg_score - predicted_class) / 3.0
        
        # Bonus for perfect alignment
        if abs(avg_score - predicted_class) < 0.5:
            consistency = min(1.0, consistency + 0.2)
        
        return max(0.0, consistency)

    def _get_recommendation(self, quality_class, confidence):
        """Generate recommendation based on prediction"""
        if confidence < 0.45:  # Adjusted threshold for balanced dataset
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
