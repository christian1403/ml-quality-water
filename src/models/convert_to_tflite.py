"""
TensorFlow Lite Model Converter for Water Quality Prediction
Converts TensorFlow models to TFLite format for serverless deployment
"""

import tensorflow as tf
import numpy as np
import os
import sys

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.data_processing.preprocessor import WaterQualityPreprocessor
from config.config import QUALITY_LABELS

class TFLiteConverter:
    """Convert TensorFlow models to TensorFlow Lite format"""
    
    def __init__(self):
        self.model_path = 'models/water_quality_model.h5'
        self.tflite_model_path = 'models/water_quality_model.tflite'
        self.preprocessor_path = 'models/water_quality_model_preprocessor.pkl'
        
    def convert_to_tflite(self, quantize=True, optimize_for_size=True):
        """
        Convert TensorFlow model to TensorFlow Lite format
        
        Args:
            quantize (bool): Apply quantization for smaller model size
            optimize_for_size (bool): Optimize for model size vs accuracy
        
        Returns:
            bool: Success status
        """
        try:
            print("🔄 Loading TensorFlow model...")
            if not os.path.exists(self.model_path):
                print(f"❌ Model file not found: {self.model_path}")
                return False
            
            # Load the TensorFlow model
            model = tf.keras.models.load_model(self.model_path)
            print(f"✅ Model loaded: {model.input_shape} -> {model.output_shape}")
            
            # Create TensorFlow Lite converter
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            
            # Set optimization flags
            if optimize_for_size:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                print("🎯 Optimization enabled: DEFAULT")
            
            # Apply quantization for smaller size
            if quantize:
                converter.target_spec.supported_types = [tf.float16]
                print("📦 Quantization enabled: FLOAT16")
            
            # Additional optimizations for serverless deployment
            converter.experimental_new_converter = True
            converter.allow_custom_ops = False
            
            print("⚙️ Converting to TensorFlow Lite...")
            tflite_model = converter.convert()
            
            # Save the TFLite model
            with open(self.tflite_model_path, 'wb') as f:
                f.write(tflite_model)
            
            # Get file sizes for comparison
            tf_size = os.path.getsize(self.model_path) / (1024 * 1024)  # MB
            tflite_size = os.path.getsize(self.tflite_model_path) / (1024 * 1024)  # MB
            compression_ratio = tf_size / tflite_size if tflite_size > 0 else 0
            
            print(f"✅ TensorFlow Lite model saved: {self.tflite_model_path}")
            print(f"📊 Size comparison:")
            print(f"   - TensorFlow model: {tf_size:.2f} MB")
            print(f"   - TensorFlow Lite: {tflite_size:.2f} MB")
            print(f"   - Compression ratio: {compression_ratio:.1f}x smaller")
            
            # Verify the converted model
            if self._verify_tflite_model(tflite_model):
                print("✅ TensorFlow Lite model verification successful")
                return True
            else:
                print("❌ TensorFlow Lite model verification failed")
                return False
                
        except Exception as e:
            print(f"❌ Error converting to TensorFlow Lite: {e}")
            return False
    
    def _verify_tflite_model(self, tflite_model):
        """
        Verify the TensorFlow Lite model works correctly
        
        Args:
            tflite_model: The TFLite model bytes
            
        Returns:
            bool: Verification success status
        """
        try:
            # Load the TFLite model
            interpreter = tf.lite.Interpreter(model_content=tflite_model)
            interpreter.allocate_tensors()
            
            # Get input and output details
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            print(f"🔍 Model verification:")
            print(f"   - Input shape: {input_details[0]['shape']}")
            print(f"   - Output shape: {output_details[0]['shape']}")
            print(f"   - Input dtype: {input_details[0]['dtype']}")
            print(f"   - Output dtype: {output_details[0]['dtype']}")
            
            # Test with sample data
            input_shape = input_details[0]['shape']
            sample_input = np.random.random(input_shape).astype(input_details[0]['dtype'])
            
            # Set input tensor
            interpreter.set_tensor(input_details[0]['index'], sample_input)
            
            # Run inference
            interpreter.invoke()
            
            # Get output
            output = interpreter.get_tensor(output_details[0]['index'])
            
            print(f"   - Sample output shape: {output.shape}")
            print(f"   - Output sum: {np.sum(output):.4f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Verification error: {e}")
            return False
    
    def create_representative_dataset(self):
        """
        Create representative dataset for quantization
        
        Returns:
            generator: Representative dataset generator
        """
        try:
            # Load preprocessor to generate realistic samples
            preprocessor = WaterQualityPreprocessor.load_preprocessor(self.preprocessor_path)
            
            def representative_data_gen():
                # Generate representative samples across the input range
                samples = [
                    [250, 1.0, 7.2],    # Excellent quality
                    [450, 3.0, 7.0],    # Good quality  
                    [750, 8.0, 6.8],    # Acceptable quality
                    [1200, 15.0, 5.5],  # Poor quality
                    [300, 2.0, 7.5],    # Edge case 1
                    [600, 4.0, 8.0],    # Edge case 2
                    [900, 10.0, 6.5],   # Edge case 3
                    [100, 0.5, 7.8],    # Low TDS
                    [2000, 20.0, 9.0],  # High values
                    [50, 25.0, 4.5],    # Extreme case
                ]
                
                for sample in samples:
                    # Preprocess the sample
                    processed = preprocessor.preprocess_single_sample(
                        sample[0], sample[1], sample[2]
                    )
                    yield [processed.astype(np.float32)]
            
            return representative_data_gen
            
        except Exception as e:
            print(f"⚠️  Could not create representative dataset: {e}")
            return None
    
    def convert_with_quantization(self):
        """
        Convert model with advanced quantization using representative dataset
        
        Returns:
            bool: Success status
        """
        try:
            print("🔄 Loading TensorFlow model for advanced quantization...")
            model = tf.keras.models.load_model(self.model_path)
            
            # Create converter
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            
            # Enable optimization
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # Set representative dataset for better quantization
            representative_dataset = self.create_representative_dataset()
            if representative_dataset:
                converter.representative_dataset = representative_dataset
                print("📊 Using representative dataset for quantization")
            
            # Use integer quantization for maximum compression
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            
            print("⚙️ Converting with INT8 quantization...")
            tflite_quantized_model = converter.convert()
            
            # Save quantized model
            quantized_path = self.tflite_model_path.replace('.tflite', '_quantized.tflite')
            with open(quantized_path, 'wb') as f:
                f.write(tflite_quantized_model)
            
            # Compare sizes
            original_size = os.path.getsize(self.model_path) / 1024  # KB
            quantized_size = os.path.getsize(quantized_path) / 1024  # KB
            
            print(f"✅ Quantized model saved: {quantized_path}")
            print(f"📊 Size comparison:")
            print(f"   - Original TF model: {original_size:.1f} KB")
            print(f"   - Quantized TFLite: {quantized_size:.1f} KB")
            print(f"   - Compression: {original_size/quantized_size:.1f}x smaller")
            
            return True
            
        except Exception as e:
            print(f"❌ Advanced quantization failed: {e}")
            print("ℹ️  Falling back to standard conversion...")
            return self.convert_to_tflite()

def main():
    """Convert existing TensorFlow model to TensorFlow Lite"""
    converter = TFLiteConverter()
    
    print("🚀 Starting TensorFlow to TensorFlow Lite conversion...")
    print("=" * 60)
    
    # Try advanced quantization first
    print("🎯 Attempting advanced quantization...")
    if converter.convert_with_quantization():
        print("✅ Advanced quantization successful!")
    else:
        print("⚠️  Advanced quantization failed, trying standard conversion...")
        
        # Fall back to standard conversion
        if converter.convert_to_tflite(quantize=True, optimize_for_size=True):
            print("✅ Standard TensorFlow Lite conversion successful!")
        else:
            print("❌ All conversion methods failed!")
            return False
    
    print("=" * 60)
    print("🎉 Model conversion completed!")
    print("📁 TensorFlow Lite models saved in 'models/' directory")
    print("🚀 Ready for serverless deployment!")
    
    return True

if __name__ == "__main__":
    main()
