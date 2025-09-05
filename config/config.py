"""
Configuration file for water quality prediction model
"""

import os
from dotenv import load_dotenv

load_dotenv()


# Model Configuration
MODEL_CONFIG = {
    'input_features': ['tds', 'turbidity', 'ph'],
    'target': 'quality',
    'model_name': 'water_quality_classifier',
    'random_state': 42,
    'test_size': 0.2,
    'validation_size': 0.2
}

# Enhanced Neural Network Architecture for Feature Engineering
NN_CONFIG = {
    'hidden_layers': [256, 128, 64, 32, 16],  # Larger architecture for more features
    'activation': 'relu',
    'dropout_rate': 0.25,  # Slightly higher dropout for regularization
    'output_activation': 'softmax',
    'optimizer': 'adam',
    'learning_rate': 0.001,
    'loss': 'categorical_crossentropy',
    'metrics': ['accuracy'],
    'epochs': 200,  # More epochs for complex feature space
    'batch_size': 64,  # Larger batch size for stability
    'early_stopping_patience': 20,
    'reduce_lr_patience': 10,
    'use_batch_normalization': True,  # Add batch norm for stability
    'l2_regularization': 0.0001  # L2 regularization
}

# Water Quality Standards (WHO/EPA guidelines)
WATER_STANDARDS = {
    'ph': {
        'excellent': (7.0, 7.5),
        'good': (6.5, 8.5),
        'poor': 'outside_good_range'
    },
    'tds': {  # mg/L
        'excellent': (0, 300),
        'good': (300, 600),
        'acceptable': (600, 900),
        'poor': (900, float('inf'))
    },
    'turbidity': {  # NTU (Nephelometric Turbidity Units)
        'excellent': (0, 1),
        'good': (1, 4),
        'acceptable': (4, 10),
        'poor': (10, float('inf'))
    }
}

# Quality Labels
QUALITY_LABELS = {
    0: 'Poor',
    1: 'Acceptable', 
    2: 'Good',
    3: 'Excellent'
}

# Data Generation Parameters
DATA_CONFIG = {
    'n_samples': 10000,
    'noise_factor': 0.1,
    'output_file': 'data/water_quality_resampled.csv'
    # 'output_file': 'data/water_quality_dataset.csv'
}

# Gemini AI Configuration
GEMINI_CONFIG = {
    'api_key': os.getenv('GEMINI_API_KEY', ''),  # Set via environment variable or fill directly
    'model_name': 'gemini-2.0-flash',
    'temperature': 0.7,
    'max_output_tokens': 500
}
