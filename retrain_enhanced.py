#!/usr/bin/env python3
"""
Retrain water quality model with advanced feature engineering
"""

import os
import sys
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.train_model import WaterQualityModel
from config.config import MODEL_CONFIG

def retrain_with_feature_engineering():
    """Retrain model with enhanced features for improved confidence"""
    print("🚀 Retraining Model with Advanced Feature Engineering")
    print("=" * 60)
    
    # Use the enhanced model with feature engineering
    model = WaterQualityModel(use_feature_engineering=True)
    
    # Load data (feature engineering will be applied automatically)
    data_path = "data/water_quality_resampled.csv"
    if not os.path.exists(data_path):
        data_path = "data/water_quality_dataset.csv"
    
    if not os.path.exists(data_path):
        print("❌ Dataset not found!")
        return False
    
    print(f"📊 Loading data from {data_path}")
    df = model.preprocessor.load_data(data_path)
    
    if df is None:
        print("❌ Failed to load data")
        return False
    
    print(f"✅ Data loaded: {df.shape}")
    print(f"📊 Original features: {MODEL_CONFIG['input_features']}")
    print(f"🔧 Engineered features: {len(model.preprocessor.engineered_feature_names)}")
    print(f"🎯 Total features: {len(MODEL_CONFIG['input_features']) + len(model.preprocessor.engineered_feature_names)}")
    
    # Train the enhanced model
    print("\n🏋️ Training enhanced model...")
    success = model.train(data_path)
    
    if success:
        # Save enhanced model with different name
        enhanced_model_path = "models/water_quality_enhanced_model.h5"
        enhanced_preprocessor_path = "models/water_quality_enhanced_model_preprocessor.pkl"
        
        model.save_model(enhanced_model_path)
        model.preprocessor.save_preprocessor(enhanced_preprocessor_path)
        
        print(f"\n✅ Enhanced model saved to {enhanced_model_path}")
        print(f"✅ Enhanced preprocessor saved to {enhanced_preprocessor_path}")
        
        # Test the enhanced model
        print("\n🧪 Testing enhanced model performance...")
        test_enhanced_model(enhanced_model_path)
        
        return True
    else:
        print("❌ Training failed")
        return False

def test_enhanced_model(model_path):
    """Test the enhanced model on sample predictions"""
    print("\n📊 Testing Enhanced Model Predictions")
    print("-" * 40)
    
    # Import predictor with enhanced model
    from src.models.predict import WaterQualityPredictor
    
    # Load enhanced model
    predictor = WaterQualityPredictor(model_path=model_path)
    
    # Test cases
    test_cases = [
        (1000, 4, 7.5, "Your original case"),
        (200, 1, 7.2, "Excellent quality"),
        (1500, 20, 5.5, "Poor quality"),
        (500, 3, 8.0, "Good quality"),
        (800, 8, 6.8, "Acceptable quality")
    ]
    
    print("Test Results:")
    for tds, turbidity, ph, description in test_cases:
        try:
            result = predictor.predict(tds, turbidity, ph)
            
            if 'error' not in result:
                confidence = result['prediction']['confidence']
                quality = result['prediction']['quality_label']
                print(f"  {description:20s} → {quality:12s} ({confidence*100:5.1f}%)")
            else:
                print(f"  {description:20s} → Error: {result['error']}")
                
        except Exception as e:
            print(f"  {description:20s} → Error: {e}")

if __name__ == "__main__":
    success = retrain_with_feature_engineering()
    
    if success:
        print("\n🎉 Enhanced model training completed successfully!")
        print("\nTo test the enhanced model:")
        print("python3 main.py --predict 1000 4 7.5 --model enhanced")
    else:
        print("\n❌ Enhanced model training failed")
        print("Please check the logs above for details")
