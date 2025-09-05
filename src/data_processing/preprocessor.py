"""
Enhanced data preprocessing utilities for water quality prediction with feature engineering
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import MODEL_CONFIG
from src.data_processing.feature_engineer import feature_engineer

class WaterQualityPreprocessor:
    """Enhanced preprocessing pipeline with advanced feature engineering"""
    
    def __init__(self, use_feature_engineering=True):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = MODEL_CONFIG['input_features']
        self.target_name = MODEL_CONFIG['target']
        self.use_feature_engineering = use_feature_engineering
        self.engineered_feature_names = []
        
    def load_data(self, filepath):
        """Load data from CSV file with optional feature engineering"""
        try:
            df = pd.read_csv(filepath)
            print(f"Data loaded successfully: {df.shape}")
            
            # Apply feature engineering if enabled
            if self.use_feature_engineering:
                print("🔧 Applying advanced feature engineering...")
                df = feature_engineer.engineer_features(df)
                self.engineered_feature_names = feature_engineer.feature_names
                print(f"Enhanced data shape: {df.shape}")
            
            return df
        except FileNotFoundError:
            print(f"File {filepath} not found. Please run data generation first.")
            return None
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def prepare_features_and_target(self, df):
        """Extract features and target from dataframe"""
        # Base features
        base_features = self.feature_names.copy()
        
        # Add engineered features if available
        if self.use_feature_engineering and self.engineered_feature_names:
            all_features = base_features + self.engineered_feature_names
            print(f"📊 Using {len(base_features)} base + {len(self.engineered_feature_names)} engineered features")
        else:
            all_features = base_features
            print(f"📊 Using {len(base_features)} base features only")
        
        # Features
        X = df[all_features].copy()
        
        # Target
        y = df[self.target_name].copy()
        
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"Target distribution:\n{pd.Series(y).value_counts().sort_index()}")
        
        return X, y
    
    def scale_features(self, X_train, X_val=None, X_test=None):
        """Scale features using StandardScaler"""
        # Fit scaler on training data
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        results = [X_train_scaled]
        
        # Transform validation and test sets (DO NOT refit!)
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            results.append(X_val_scaled)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            results.append(X_test_scaled)
        
        return results if len(results) > 1 else results[0]
    
    def encode_labels(self, y_train, y_val=None, y_test=None):
        """Convert labels to categorical format for neural network"""
        # Convert to categorical
        y_train_cat = to_categorical(y_train)
        
        results = [y_train_cat]
        
        if y_val is not None:
            y_val_cat = to_categorical(y_val)
            results.append(y_val_cat)
        
        if y_test is not None:
            y_test_cat = to_categorical(y_test)
            results.append(y_test_cat)
        
        return results if len(results) > 1 else results[0]
    
    def split_data(self, X, y, test_size=None, val_size=None, random_state=None):
        """Split data into train, validation, and test sets"""
        test_size = test_size or MODEL_CONFIG['test_size']
        val_size = val_size or MODEL_CONFIG['validation_size']
        random_state = random_state or MODEL_CONFIG['random_state']
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: separate train and validation from remaining data
        val_size_adjusted = val_size / (1 - test_size)  # Adjust validation size
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=random_state, stratify=y_temp
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def calculate_class_weights(self, y_train):
        """Calculate class weights for imbalanced dataset"""
        class_weights = class_weight.compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        
        class_weight_dict = dict(enumerate(class_weights))
        print(f"Class weights: {class_weight_dict}")
        
        return class_weight_dict
    
    def preprocess_pipeline(self, df):
        """Complete preprocessing pipeline"""
        print("=== Starting Data Preprocessing ===")
        
        # 1. Prepare features and target
        X, y = self.prepare_features_and_target(df)
        
        # 2. Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        
        # 3. Scale features
        X_train_scaled, X_val_scaled, X_test_scaled = self.scale_features(
            X_train, X_val, X_test
        )
        
        # 4. Encode labels for neural network
        y_train_cat, y_val_cat, y_test_cat = self.encode_labels(
            y_train, y_val, y_test
        )
        
        # 5. Calculate class weights
        class_weights = self.calculate_class_weights(y_train)
        
        print("=== Preprocessing Complete ===")
        
        return {
            'X_train': X_train_scaled,
            'X_val': X_val_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train_cat,
            'y_val': y_val_cat,
            'y_test': y_test_cat,
            'y_train_raw': y_train,
            'y_val_raw': y_val,
            'y_test_raw': y_test,
            'class_weights': class_weights
        }
    
    def preprocess_single_sample(self, tds, turbidity, ph):
        """Preprocess a single sample for prediction"""
        # Create dataframe with feature names
        sample_df = pd.DataFrame({
            'tds': [tds],
            'turbidity': [turbidity], 
            'ph': [ph]
        })
        
        # Scale using fitted scaler (DO NOT refit!)
        sample_scaled = self.scaler.transform(sample_df)
        
        return sample_scaled
    
    def preprocess_single_sample(self, tds, turbidity, ph):
        """
        Preprocess a single sample for prediction (with feature engineering support)
        
        Args:
            tds: Total Dissolved Solids
            turbidity: Turbidity
            ph: pH level
            
        Returns:
            Scaled feature array ready for prediction
        """
        # Create DataFrame for the sample
        sample_data = pd.DataFrame({
            'tds': [tds],
            'turbidity': [turbidity], 
            'ph': [ph]
        })
        
        # Apply feature engineering if enabled
        if self.use_feature_engineering:
            sample_data = feature_engineer.engineer_features(sample_data)
        
        # Get all required features
        if self.use_feature_engineering and self.engineered_feature_names:
            all_features = self.feature_names + self.engineered_feature_names
        else:
            all_features = self.feature_names
        
        # Select features and scale
        sample_features = sample_data[all_features].values
        sample_scaled = self.scaler.transform(sample_features)
        
        return sample_scaled
    
    def save_preprocessor(self, filepath):
        """Save the preprocessor to file"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Preprocessor saved to {filepath}")
    
    @classmethod
    def load_preprocessor(cls, filepath):
        """Load preprocessor from file"""
        import pickle
        with open(filepath, 'rb') as f:
            preprocessor = pickle.load(f)
        print(f"Preprocessor loaded from {filepath}")
        return preprocessor

def main():
    """Test preprocessing pipeline"""
    preprocessor = WaterQualityPreprocessor()
    
    # Load data
    # df = preprocessor.load_data('data/water_quality_dataset.csv')
    df = preprocessor.load_data('data/water_quality_resampled.csv')
    if df is None:
        print("Please run data generation first: python src/data_processing/generate_data.py")
        return
    
    # Run preprocessing pipeline
    processed_data = preprocessor.preprocess_pipeline(df)
    
    print("\n=== Processed Data Shapes ===")
    print(f"X_train: {processed_data['X_train'].shape}")
    print(f"X_val: {processed_data['X_val'].shape}")
    print(f"X_test: {processed_data['X_test'].shape}")
    print(f"y_train: {processed_data['y_train'].shape}")
    
    return processed_data

if __name__ == "__main__":
    main()
