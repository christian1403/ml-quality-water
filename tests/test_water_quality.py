"""
Unit tests for water quality prediction system
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing.generate_data import WaterQualityDataGenerator
from src.data_processing.preprocessor import WaterQualityPreprocessor
from src.utils.analysis_utils import validate_sensor_reading, calculate_water_quality_index

class TestWaterQualityDataGenerator(unittest.TestCase):
    """Test data generation functionality"""
    
    def setUp(self):
        self.generator = WaterQualityDataGenerator(n_samples=100, random_state=42)
    
    def test_generate_realistic_sample(self):
        """Test realistic sample generation"""
        tds, turbidity, ph = self.generator.generate_realistic_sample(quality_target=3)
        
        # Check value ranges
        self.assertTrue(50 <= tds <= 3000)
        self.assertTrue(0.1 <= turbidity <= 50)
        self.assertTrue(4.0 <= ph <= 12.0)
    
    def test_generate_dataset(self):
        """Test complete dataset generation"""
        df = self.generator.generate_dataset()
        
        # Check dataset properties
        self.assertEqual(len(df), 100)
        self.assertTrue(all(col in df.columns for col in ['tds', 'turbidity', 'ph', 'quality']))
        self.assertTrue(df['quality'].min() >= 0)
        self.assertTrue(df['quality'].max() <= 3)

class TestWaterQualityPreprocessor(unittest.TestCase):
    """Test preprocessing functionality"""
    
    def setUp(self):
        self.preprocessor = WaterQualityPreprocessor()
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'tds': [250, 500, 800, 1200],
            'turbidity': [0.5, 2.0, 5.0, 15.0],
            'ph': [7.2, 7.0, 6.8, 5.5],
            'quality': [3, 2, 1, 0]
        })
    
    def test_prepare_features_and_target(self):
        """Test feature and target preparation"""
        X, y = self.preprocessor.prepare_features_and_target(self.sample_data)
        
        self.assertEqual(X.shape[1], 3)  # 3 features
        self.assertEqual(len(X), len(y))
        self.assertTrue(all(col in X.columns for col in ['tds', 'turbidity', 'ph']))
    
    def test_scale_features(self):
        """Test feature scaling"""
        X, y = self.preprocessor.prepare_features_and_target(self.sample_data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
        
        X_train_scaled, X_test_scaled = self.preprocessor.scale_features(X_train, X_test)
        
        # Check that scaling was applied
        self.assertAlmostEqual(np.mean(X_train_scaled), 0, places=10)
        self.assertAlmostEqual(np.std(X_train_scaled), 1, places=1)

class TestAnalysisUtils(unittest.TestCase):
    """Test analysis utility functions"""
    
    def test_validate_sensor_reading(self):
        """Test sensor reading validation"""
        # Valid readings
        result = validate_sensor_reading(tds=300, turbidity=2.0, ph=7.0)
        self.assertTrue(result['valid'])
        self.assertEqual(len(result['errors']), 0)
        
        # Invalid pH
        result = validate_sensor_reading(ph=15.0)
        self.assertFalse(result['valid'])
        self.assertTrue(len(result['errors']) > 0)
        
        # Invalid TDS
        result = validate_sensor_reading(tds=-100)
        self.assertFalse(result['valid'])
        self.assertTrue(len(result['errors']) > 0)
    
    def test_calculate_water_quality_index(self):
        """Test WQI calculation"""
        # Excellent water
        wqi = calculate_water_quality_index(tds=200, turbidity=0.5, ph=7.0)
        self.assertTrue(80 <= wqi <= 100)
        
        # Poor water
        wqi = calculate_water_quality_index(tds=1500, turbidity=20, ph=5.0)
        self.assertTrue(0 <= wqi <= 40)

class TestEndToEndPipeline(unittest.TestCase):
    """Test complete pipeline functionality"""
    
    def test_data_generation_to_preprocessing(self):
        """Test complete pipeline from data generation to preprocessing"""
        # Generate small dataset
        generator = WaterQualityDataGenerator(n_samples=50, random_state=42)
        df = generator.generate_dataset()
        
        # Preprocess data
        preprocessor = WaterQualityPreprocessor()
        X, y = preprocessor.prepare_features_and_target(df)
        
        # Check pipeline results
        self.assertEqual(len(X), 50)
        self.assertEqual(len(y), 50)
        self.assertTrue(all(col in X.columns for col in ['tds', 'turbidity', 'ph']))

def run_tests():
    """Run all tests"""
    print(\"Running water quality prediction system tests...\")\n    \n    unittest.main(argv=[''], exit=False, verbosity=2)\n    \n    print(\"\\nAll tests completed!\")\n\nif __name__ == \"__main__\":\n    run_tests()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "test_main",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "run_tests()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
