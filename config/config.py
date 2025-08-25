"""
Configuration file for water quality prediction model
"""

# Model Configuration
MODEL_CONFIG = {
    'input_features': ['tds', 'turbidity', 'ph'],
    'target': 'quality',
    'model_name': 'water_quality_classifier',
    'random_state': 42,
    'test_size': 0.2,
    'validation_size': 0.2
}

# Neural Network Architecture
NN_CONFIG = {
    'hidden_layers': [64, 32, 16],
    'activation': 'relu',
    'dropout_rate': 0.3,
    'output_activation': 'softmax',
    'optimizer': 'adam',
    'loss': 'categorical_crossentropy',
    'metrics': ['accuracy'],
    'epochs': 100,
    'batch_size': 32,
    'early_stopping_patience': 10
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
    'output_file': 'data/water_quality_dataset.csv'
}
