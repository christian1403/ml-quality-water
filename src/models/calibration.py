"""
Advanced Model Calibration for Water Quality Prediction
Implements temperature scaling and isotonic regression for better confidence calibration
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
import joblib
import os

class ConfidenceCalibrator:
    """
    Advanced confidence calibration using multiple techniques:
    1. Temperature Scaling - Single parameter optimization
    2. Platt Scaling - Sigmoid function fitting
    3. Isotonic Regression - Non-parametric calibration
    """
    
    def __init__(self, method='temperature'):
        """
        Initialize calibrator
        
        Args:
            method: 'temperature', 'platt', 'isotonic', or 'ensemble'
        """
        self.method = method
        self.temperature = 1.0
        self.calibrator = None
        self.is_fitted = False
        
    def fit(self, logits, true_labels, validation_split=0.2):
        """
        Fit calibration parameters using validation data
        
        Args:
            logits: Raw model outputs (pre-softmax)
            true_labels: True class labels
            validation_split: Fraction of data to use for calibration
        """
        # Convert to numpy arrays
        logits = np.array(logits)
        true_labels = np.array(true_labels)
        
        # Split data for calibration
        n_val = int(len(logits) * validation_split)
        if n_val < 100:  # Ensure minimum validation size
            n_val = min(100, len(logits) // 2)
            
        indices = np.random.permutation(len(logits))
        val_indices = indices[:n_val]
        
        val_logits = logits[val_indices]
        val_labels = true_labels[val_indices]
        
        if self.method == 'temperature':
            self._fit_temperature_scaling(val_logits, val_labels)
        elif self.method == 'platt':
            self._fit_platt_scaling(val_logits, val_labels)
        elif self.method == 'isotonic':
            self._fit_isotonic_regression(val_logits, val_labels)
        elif self.method == 'ensemble':
            self._fit_ensemble_calibration(val_logits, val_labels)
        
        self.is_fitted = True
        return self
    
    def _fit_temperature_scaling(self, logits, true_labels):
        """Fit temperature scaling parameter"""
        def temperature_scale_loss(temperature):
            scaled_logits = logits / max(temperature, 0.01)  # Prevent division by 0
            probs = self._softmax(scaled_logits)
            
            # Negative log-likelihood
            epsilon = 1e-15  # Prevent log(0)
            probs = np.clip(probs, epsilon, 1 - epsilon)
            
            nll = -np.sum(np.log(probs[range(len(true_labels)), true_labels]))
            return nll
        
        # Optimize temperature parameter
        result = minimize_scalar(
            temperature_scale_loss, 
            bounds=(0.1, 10.0), 
            method='bounded'
        )
        
        self.temperature = result.x
        print(f"🔥 Temperature Scaling fitted: T = {self.temperature:.3f}")
    
    def _fit_platt_scaling(self, logits, true_labels):
        """Fit Platt scaling (sigmoid) calibration"""
        from sklearn.linear_model import LogisticRegression
        
        # Get max probabilities as confidence scores
        probs = self._softmax(logits)
        max_probs = np.max(probs, axis=1).reshape(-1, 1)
        
        # Binary calibration: confident vs not confident
        binary_labels = (np.max(probs, axis=1) == probs[range(len(true_labels)), true_labels]).astype(int)
        
        self.calibrator = LogisticRegression()
        self.calibrator.fit(max_probs, binary_labels)
        print("📈 Platt Scaling fitted successfully")
    
    def _fit_isotonic_regression(self, logits, true_labels):
        """Fit isotonic regression calibration"""
        probs = self._softmax(logits)
        max_probs = np.max(probs, axis=1)
        
        # Binary calibration target
        binary_labels = (np.argmax(probs, axis=1) == true_labels).astype(float)
        
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.calibrator.fit(max_probs, binary_labels)
        print("📊 Isotonic Regression fitted successfully")
    
    def _fit_ensemble_calibration(self, logits, true_labels):
        """Fit ensemble of calibration methods"""
        # Fit all methods
        self._fit_temperature_scaling(logits, true_labels)
        
        # Store temperature for ensemble
        temp_param = self.temperature
        
        # Fit other methods
        self._fit_platt_scaling(logits, true_labels)
        platt_calibrator = self.calibrator
        
        self._fit_isotonic_regression(logits, true_labels)
        isotonic_calibrator = self.calibrator
        
        # Store all calibrators
        self.calibrator = {
            'temperature': temp_param,
            'platt': platt_calibrator,
            'isotonic': isotonic_calibrator
        }
        print("🎯 Ensemble Calibration fitted successfully")
    
    def calibrate_probabilities(self, logits):
        """
        Apply calibration to raw logits
        
        Args:
            logits: Raw model outputs
            
        Returns:
            Calibrated probabilities and enhanced confidence
        """
        if not self.is_fitted:
            print("⚠️  Warning: Calibrator not fitted. Using original probabilities.")
            return self._softmax(logits)
        
        logits = np.array(logits)
        
        if self.method == 'temperature':
            return self._apply_temperature_scaling(logits)
        elif self.method == 'platt':
            return self._apply_platt_scaling(logits)
        elif self.method == 'isotonic':
            return self._apply_isotonic_scaling(logits)
        elif self.method == 'ensemble':
            return self._apply_ensemble_scaling(logits)
        
        return self._softmax(logits)
    
    def _apply_temperature_scaling(self, logits):
        """Apply temperature scaling"""
        scaled_logits = logits / self.temperature
        calibrated_probs = self._softmax(scaled_logits)
        return calibrated_probs
    
    def _apply_platt_scaling(self, logits):
        """Apply Platt scaling"""
        probs = self._softmax(logits)
        max_probs = np.max(probs, axis=1).reshape(-1, 1)
        
        # Get calibration confidence
        calibrated_confidence = self.calibrator.predict_proba(max_probs)[:, 1]
        
        # Adjust probabilities based on calibrated confidence
        adjusted_probs = probs.copy()
        for i in range(len(probs)):
            pred_class = np.argmax(probs[i])
            confidence_adjustment = calibrated_confidence[i] / np.max(probs[i])
            adjusted_probs[i] *= confidence_adjustment
            adjusted_probs[i] /= np.sum(adjusted_probs[i])  # Renormalize
        
        return adjusted_probs
    
    def _apply_isotonic_scaling(self, logits):
        """Apply isotonic regression scaling"""
        probs = self._softmax(logits)
        max_probs = np.max(probs, axis=1)
        
        # Get calibrated confidence
        calibrated_confidence = self.calibrator.predict(max_probs)
        
        # Adjust probabilities
        adjusted_probs = probs.copy()
        for i in range(len(probs)):
            pred_class = np.argmax(probs[i])
            confidence_adjustment = calibrated_confidence[i] / np.max(probs[i])
            adjusted_probs[i] *= confidence_adjustment
            adjusted_probs[i] /= np.sum(adjusted_probs[i])  # Renormalize
        
        return adjusted_probs
    
    def _apply_ensemble_scaling(self, logits):
        """Apply ensemble calibration"""
        # Temperature scaling
        temp_probs = self._apply_temperature_scaling(logits)
        
        # Platt scaling
        self.calibrator = self.calibrator['platt']
        platt_probs = self._apply_platt_scaling(logits)
        
        # Isotonic scaling
        self.calibrator = self.calibrator['isotonic']
        isotonic_probs = self._apply_isotonic_scaling(logits)
        
        # Weighted ensemble (temperature gets highest weight)
        ensemble_probs = (0.5 * temp_probs + 
                         0.3 * platt_probs + 
                         0.2 * isotonic_probs)
        
        return ensemble_probs
    
    def _softmax(self, logits):
        """Stable softmax computation"""
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    def save(self, filepath):
        """Save calibrator to file"""
        calibrator_data = {
            'method': self.method,
            'temperature': self.temperature,
            'calibrator': self.calibrator,
            'is_fitted': self.is_fitted
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(calibrator_data, filepath)
        print(f"💾 Calibrator saved to {filepath}")
    
    def load(self, filepath):
        """Load calibrator from file"""
        if os.path.exists(filepath):
            calibrator_data = joblib.load(filepath)
            self.method = calibrator_data['method']
            self.temperature = calibrator_data['temperature']
            self.calibrator = calibrator_data['calibrator']
            self.is_fitted = calibrator_data['is_fitted']
            print(f"📁 Calibrator loaded from {filepath}")
            return True
        return False

def create_calibration_dataset():
    """
    Create calibration dataset from existing predictions
    This will be used to fit the calibrator
    """
    print("🔧 Creating calibration dataset...")
    
    # Load your existing model and data
    from src.models.predict import WaterQualityPredictor
    import pandas as pd
    
    # Load test data
    data_path = "data/water_quality_resampled.csv"
    if not os.path.exists(data_path):
        data_path = "data/water_quality_dataset.csv"
    
    df = pd.read_csv(data_path)
    
    # Split for calibration (use 20% of data)
    cal_size = min(2000, len(df) // 5)  # Max 2000 samples for calibration
    cal_data = df.sample(n=cal_size, random_state=42)
    
    predictor = WaterQualityPredictor()
    
    logits_list = []
    labels_list = []
    
    print(f"🔄 Processing {len(cal_data)} samples for calibration...")
    
    for idx, row in cal_data.iterrows():
        # Get raw model output (before softmax)
        try:
            # This would be your model's raw output
            result = predictor.predict_single(row['tds'], row['turbidity'], row['ph'])
            
            if 'error' not in result:
                # Convert probabilities back to logits (approximate)
                probs = [
                    float(result['probabilities']['Poor'].strip('%')) / 100,
                    float(result['probabilities']['Acceptable'].strip('%')) / 100,
                    float(result['probabilities']['Good'].strip('%')) / 100,
                    float(result['probabilities']['Excellent'].strip('%')) / 100
                ]
                
                # Convert to logits (inverse softmax)
                epsilon = 1e-15
                probs = np.clip(probs, epsilon, 1 - epsilon)
                logits = np.log(probs)
                
                logits_list.append(logits)
                labels_list.append(row['quality'])
                
        except Exception as e:
            continue
    
    return np.array(logits_list), np.array(labels_list)

if __name__ == "__main__":
    # Test calibration
    print("🧪 Testing Confidence Calibration...")
    
    # Create calibration data
    logits, labels = create_calibration_dataset()
    
    if len(logits) > 0:
        # Fit calibrator
        calibrator = ConfidenceCalibrator(method='temperature')
        calibrator.fit(logits, labels)
        
        # Save calibrator
        calibrator.save('models/confidence_calibrator.pkl')
        
        print("✅ Calibration completed successfully!")
    else:
        print("❌ Could not create calibration dataset")
