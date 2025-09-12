"""
Convert TensorFlow H5 models to TensorFlow Lite format for optimized inference
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing.preprocessor import WaterQualityPreprocessor

class ModelToTFLiteConverter:
    """Convert H5 models to optimized TFLite format"""
    
    def __init__(self):
        self.models_dir = 'models'
        self.h5_models = [
            'water_quality_model.h5',
            'best_water_quality_model.h5', 
            'water_quality_enhanced_model.h5'
        ]
    
    def convert_model_to_tflite(self, h5_model_path, tflite_model_path, optimize=True):
        """
        Convert a single H5 model to TFLite format
        
        Args:
            h5_model_path (str): Path to H5 model file
            tflite_model_path (str): Output path for TFLite model
            optimize (bool): Whether to apply optimizations
        
        Returns:
            bool: Success status
        """
        try:
            print(f"🔄 Converting {h5_model_path} to TFLite...")
            
            # Load the H5 model
            model = tf.keras.models.load_model(h5_model_path)
            print(f"✅ Loaded H5 model: {h5_model_path}")
            
            # Create TFLite converter
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            
            if optimize:
                # Apply optimizations for smaller model size and faster inference
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                
                # Set representative dataset for quantization
                converter.representative_dataset = self._get_representative_dataset
                
                # Optional: Use float16 quantization for smaller models
                converter.target_spec.supported_types = [tf.float16]
                
                print("🚀 Applying optimizations (quantization, float16)...")
            
            # Convert the model
            tflite_model = converter.convert()
            
            # Save the TFLite model
            with open(tflite_model_path, 'wb') as f:
                f.write(tflite_model)
            
            # Get file sizes for comparison
            h5_size = os.path.getsize(h5_model_path) / (1024 * 1024)  # MB
            tflite_size = os.path.getsize(tflite_model_path) / (1024 * 1024)  # MB
            compression_ratio = (h5_size - tflite_size) / h5_size * 100
            
            print(f"✅ Successfully converted to: {tflite_model_path}")
            print(f"📊 Size comparison:")
            print(f"   - Original H5: {h5_size:.2f} MB")
            print(f"   - TFLite: {tflite_size:.2f} MB")
            print(f"   - Compression: {compression_ratio:.1f}% smaller")
            
            # Validate the converted model
            self._validate_tflite_model(tflite_model_path, model)
            
            return True
            
        except Exception as e:
            print(f"❌ Error converting {h5_model_path}: {e}")
            return False
    
    def _get_representative_dataset(self):
        """
        Generate representative dataset for quantization
        This helps TFLite optimize the model based on typical input ranges
        """
        # Load sample data for representative dataset
        try:
            # Load the actual dataset if available
            data_path = 'data/water_quality_resampled.csv'
            if os.path.exists(data_path):
                df = pd.read_csv(data_path)
                # Get input features
                X = df[['tds', 'turbidity', 'ph']].values
                
                # Load preprocessor to get the same preprocessing
                preprocessor_path = 'models/water_quality_model_preprocessor.pkl'
                if os.path.exists(preprocessor_path):
                    preprocessor = WaterQualityPreprocessor.load_preprocessor(preprocessor_path)
                    X_processed = preprocessor.preprocessor.transform(X)
                else:
                    # Fallback: simple standardization
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    X_processed = scaler.fit_transform(X)
                
                # Take a representative sample
                sample_size = min(100, len(X_processed))
                indices = np.random.choice(len(X_processed), sample_size, replace=False)
                representative_data = X_processed[indices]
                
            else:
                # Generate synthetic representative data if no dataset available
                print("📝 Generating synthetic representative dataset...")
                representative_data = self._generate_synthetic_data()
            
            # Convert to generator for TFLite
            for sample in representative_data:
                yield [sample.astype(np.float32).reshape(1, -1)]
                
        except Exception as e:
            print(f"⚠️  Warning: Could not load representative dataset: {e}")
            print("📝 Using synthetic data...")
            # Fallback to synthetic data
            representative_data = self._generate_synthetic_data()
            for sample in representative_data:
                yield [sample.astype(np.float32).reshape(1, -1)]
    
    def _generate_synthetic_data(self):
        """Generate synthetic water quality data for representative dataset"""
        np.random.seed(42)
        
        # Generate realistic water quality parameter ranges
        tds_samples = np.random.uniform(50, 1500, 100)      # TDS: 50-1500 mg/L
        turbidity_samples = np.random.uniform(0.1, 50, 100) # Turbidity: 0.1-50 NTU  
        ph_samples = np.random.uniform(5.0, 10.0, 100)      # pH: 5.0-10.0
        
        # Combine into samples
        synthetic_data = np.column_stack([tds_samples, turbidity_samples, ph_samples])
        
        # Apply basic standardization (approximate)
        synthetic_data = (synthetic_data - np.mean(synthetic_data, axis=0)) / np.std(synthetic_data, axis=0)
        
        return synthetic_data
    
    def _validate_tflite_model(self, tflite_model_path, original_model):
        """
        Validate that the TFLite model produces similar outputs to the original model
        """
        try:
            print("🔍 Validating TFLite model...")
            
            # Load TFLite model
            interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
            interpreter.allocate_tensors()
            
            # Get input and output details
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            print(f"📋 TFLite Model Details:")
            print(f"   - Input shape: {input_details[0]['shape']}")
            print(f"   - Input type: {input_details[0]['dtype']}")
            print(f"   - Output shape: {output_details[0]['shape']}")
            print(f"   - Output type: {output_details[0]['dtype']}")
            
            # Test with a sample input
            test_input = np.array([[300.0, 2.5, 7.0]], dtype=np.float32)  # Sample water quality data
            
            # Original model prediction
            original_pred = original_model.predict(test_input, verbose=0)
            
            # TFLite model prediction
            interpreter.set_tensor(input_details[0]['index'], test_input)
            interpreter.invoke()
            tflite_pred = interpreter.get_tensor(output_details[0]['index'])
            
            # Compare predictions
            mse = np.mean((original_pred - tflite_pred) ** 2)
            max_diff = np.max(np.abs(original_pred - tflite_pred))
            
            print(f"📊 Validation Results:")
            print(f"   - MSE between models: {mse:.6f}")
            print(f"   - Max difference: {max_diff:.6f}")
            
            if mse < 0.001 and max_diff < 0.01:
                print("✅ TFLite model validation PASSED - outputs are very similar")
            elif mse < 0.01 and max_diff < 0.1:
                print("⚠️  TFLite model validation PASSED with minor differences (acceptable)")
            else:
                print("❌ TFLite model validation FAILED - significant differences detected")
                print(f"   Original: {original_pred}")
                print(f"   TFLite:   {tflite_pred}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error validating TFLite model: {e}")
            return False
    
    def convert_all_models(self):
        """Convert all H5 models to TFLite format"""
        print("🚀 Starting batch conversion of H5 models to TFLite...")
        print("="*60)
        
        success_count = 0
        total_count = len(self.h5_models)
        
        for h5_model in self.h5_models:
            h5_path = os.path.join(self.models_dir, h5_model)
            tflite_path = os.path.join(self.models_dir, h5_model.replace('.h5', '.tflite'))
            
            if not os.path.exists(h5_path):
                print(f"⚠️  Skipping {h5_model} - file not found")
                continue
            
            print(f"\n📦 Converting {h5_model}...")
            if self.convert_model_to_tflite(h5_path, tflite_path):
                success_count += 1
            
            print("-" * 40)
        
        print(f"\n🎯 Conversion Summary:")
        print(f"   - Successfully converted: {success_count}/{total_count}")
        print(f"   - TFLite models saved in: {self.models_dir}/")
        
        if success_count > 0:
            print("\n📁 Available TFLite models:")
            for h5_model in self.h5_models:
                tflite_path = os.path.join(self.models_dir, h5_model.replace('.h5', '.tflite'))
                if os.path.exists(tflite_path):
                    size = os.path.getsize(tflite_path) / (1024 * 1024)
                    print(f"   ✅ {h5_model.replace('.h5', '.tflite')} ({size:.2f} MB)")
        
        return success_count == total_count

def main():
    """Main conversion script"""
    print("🔧 TensorFlow H5 to TFLite Model Converter")
    print("=" * 50)
    
    converter = ModelToTFLiteConverter()
    
    # Convert all models
    success = converter.convert_all_models()
    
    if success:
        print("\n🎉 All models converted successfully!")
        print("💡 Next steps:")
        print("   1. Update predict.py to use TFLite models")
        print("   2. Test the TFLite models with sample predictions")
        print("   3. Update requirements.txt if needed")
    else:
        print("\n⚠️  Some conversions failed. Check the error messages above.")
    
    return success

if __name__ == "__main__":
    main()
