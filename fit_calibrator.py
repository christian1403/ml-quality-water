#!/usr/bin/env python3
"""
Fit confidence calibrator for improved water quality predictions
This script creates and trains a calibrator to improve prediction confidence
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.predict import WaterQualityPredictor
from src.models.calibration import ConfidenceCalibrator

def collect_calibration_data(n_samples=2000):
    """
    Collect model predictions for calibration
    
    Args:
        n_samples: Number of samples to use for calibration
    
    Returns:
        logits, true_labels: Arrays for calibration
    """
    print("🔄 Collecting calibration data...")
    
    # Load dataset
    data_path = "data/water_quality_resampled.csv"
    if not os.path.exists(data_path):
        data_path = "data/water_quality_dataset.csv"
    
    if not os.path.exists(data_path):
        print("❌ Dataset not found!")
        return None, None
    
    df = pd.read_csv(data_path)
    print(f"📊 Loaded dataset with {len(df)} samples")
    
    # Sample data for calibration
    if len(df) > n_samples:
        cal_data = df.sample(n=n_samples, random_state=42)
    else:
        cal_data = df
    
    print(f"🎯 Using {len(cal_data)} samples for calibration")
    
    # Initialize predictor
    predictor = WaterQualityPredictor()
    
    if predictor.model is None:
        print("❌ Model not loaded. Please train the model first.")
        return None, None
    
    logits_list = []
    labels_list = []
    
    print("🔄 Processing samples...")
    processed = 0
    
    for idx, row in cal_data.iterrows():
        try:
            # Get raw model predictions (before any enhancement)
            sample_scaled = predictor.preprocessor.preprocess_single_sample(
                row['tds'], row['turbidity'], row['ph']
            )
            
            # Get raw probabilities
            pred_proba = predictor.model.predict(sample_scaled, verbose=0)
            
            # Convert to logits
            epsilon = 1e-15
            pred_proba_clipped = np.clip(pred_proba[0], epsilon, 1 - epsilon)
            logits = np.log(pred_proba_clipped)
            
            logits_list.append(logits)
            labels_list.append(row['quality'])
            
            processed += 1
            if processed % 500 == 0:
                print(f"   Processed {processed}/{len(cal_data)} samples")
                
        except Exception as e:
            print(f"   ⚠️  Error processing sample {idx}: {e}")
            continue
    
    print(f"✅ Successfully processed {len(logits_list)} samples")
    return np.array(logits_list), np.array(labels_list)

def fit_calibrator():
    """Fit and save confidence calibrator"""
    print("🚀 Starting Confidence Calibration Process...")
    print("=" * 50)
    
    # Collect calibration data
    logits, labels = collect_calibration_data(2000)
    
    if logits is None or len(logits) < 100:
        print("❌ Insufficient data for calibration")
        return False
    
    print(f"\n📊 Calibration Dataset Statistics:")
    print(f"   Samples: {len(logits)}")
    print(f"   Classes: {np.unique(labels)}")
    print(f"   Class distribution: {np.bincount(labels)}")
    
    # Split data for calibration fitting and testing
    train_logits, test_logits, train_labels, test_labels = train_test_split(
        logits, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    print(f"\n🔧 Training calibrator on {len(train_logits)} samples...")
    
    # Test different calibration methods
    methods = ['temperature', 'platt', 'isotonic']
    best_method = None
    best_score = -np.inf
    
    for method in methods:
        print(f"\n🧪 Testing {method.upper()} calibration...")
        
        try:
            # Fit calibrator
            calibrator = ConfidenceCalibrator(method=method)
            calibrator.fit(train_logits, train_labels, validation_split=0.0)
            
            # Test calibration quality
            test_proba_original = np.exp(test_logits) / np.sum(np.exp(test_logits), axis=1, keepdims=True)
            test_proba_calibrated = calibrator.calibrate_probabilities(test_logits)
            
            # Calculate improvement metrics
            original_confidence = np.max(test_proba_original, axis=1)
            calibrated_confidence = np.max(test_proba_calibrated, axis=1)
            
            # Accuracy improvement
            original_predictions = np.argmax(test_proba_original, axis=1)
            calibrated_predictions = np.argmax(test_proba_calibrated, axis=1)
            
            original_accuracy = np.mean(original_predictions == test_labels)
            calibrated_accuracy = np.mean(calibrated_predictions == test_labels)
            
            # Confidence improvement
            avg_confidence_improvement = np.mean(calibrated_confidence - original_confidence)
            
            score = calibrated_accuracy + avg_confidence_improvement * 0.5
            
            print(f"   📈 Results for {method}:")
            print(f"      Original Accuracy: {original_accuracy:.4f}")
            print(f"      Calibrated Accuracy: {calibrated_accuracy:.4f}")
            print(f"      Avg Confidence Change: {avg_confidence_improvement:+.4f}")
            print(f"      Combined Score: {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_method = method
                best_calibrator = calibrator
                
        except Exception as e:
            print(f"   ❌ Error with {method}: {e}")
            continue
    
    if best_method is None:
        print("❌ No calibration method succeeded")
        return False
    
    print(f"\n🏆 Best method: {best_method.upper()}")
    print(f"   Score: {best_score:.4f}")
    
    # Save the best calibrator
    os.makedirs('models', exist_ok=True)
    best_calibrator.save('models/confidence_calibrator.pkl')
    
    print("\n✅ Calibration completed successfully!")
    print("🎯 Calibrator saved to models/confidence_calibrator.pkl")
    print("\nNow test improved confidence with:")
    print("python3 main.py --predict 1000 4 7.5")
    
    return True

def test_calibration_improvement():
    """Test the improvement before and after calibration"""
    print("\n🧪 Testing Calibration Improvement...")
    
    # Test cases
    test_cases = [
        (1000, 4, 7.5),   # Your original case
        (200, 1, 7.2),    # Excellent quality
        (1500, 20, 5.5),  # Poor quality
        (500, 3, 8.0),    # Good quality
        (800, 8, 6.8)     # Acceptable quality
    ]
    
    # Test without calibrator
    print("\n📊 WITHOUT Calibration:")
    print("-" * 40)
    
    predictor_no_cal = WaterQualityPredictor()
    predictor_no_cal.calibrator = None  # Disable calibrator
    
    results_no_cal = []
    for tds, turbidity, ph in test_cases:
        result = predictor_no_cal.predict_single(tds, turbidity, ph)
        confidence = result.get('confidence', 0)
        results_no_cal.append(confidence)
        print(f"TDS={tds:4d}, Turb={turbidity:2.1f}, pH={ph:3.1f} → {confidence:5.1f}%")
    
    # Test with calibrator
    print("\n📊 WITH Calibration:")
    print("-" * 40)
    
    predictor_with_cal = WaterQualityPredictor()
    
    results_with_cal = []
    for tds, turbidity, ph in test_cases:
        result = predictor_with_cal.predict_single(tds, turbidity, ph)
        confidence = result.get('confidence', 0)
        results_with_cal.append(confidence)
        print(f"TDS={tds:4d}, Turb={turbidity:2.1f}, pH={ph:3.1f} → {confidence:5.1f}%")
    
    # Calculate improvements
    print("\n📈 Confidence Improvements:")
    print("-" * 40)
    
    for i, (tds, turbidity, ph) in enumerate(test_cases):
        improvement = results_with_cal[i] - results_no_cal[i]
        print(f"TDS={tds:4d}, Turb={turbidity:2.1f}, pH={ph:3.1f} → {improvement:+5.1f}%")
    
    avg_improvement = np.mean([results_with_cal[i] - results_no_cal[i] for i in range(len(test_cases))])
    print(f"\n🎯 Average Improvement: {avg_improvement:+.1f}%")

if __name__ == "__main__":
    print("🚀 Advanced Confidence Calibration System")
    print("=" * 50)
    
    # Fit calibrator
    success = fit_calibrator()
    
    if success:
        # Test improvement
        test_calibration_improvement()
    else:
        print("\n❌ Calibration failed. Please check your model and data.")
