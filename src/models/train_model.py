"""
TensorFlow neural network model for water quality classification
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import NN_CONFIG, QUALITY_LABELS, MODEL_CONFIG
from data_processing.preprocessor import WaterQualityPreprocessor

class WaterQualityModel:
    """Neural Network model for water quality classification"""
    
    def __init__(self):
        self.model = None
        self.history = None
        self.preprocessor = WaterQualityPreprocessor()
        
    def build_model(self, input_dim=3, num_classes=4):
        """Build neural network architecture"""
        model = Sequential([
            # Input layer with L2 regularization
            Dense(NN_CONFIG['hidden_layers'][0], 
                  input_dim=input_dim,
                  activation=NN_CONFIG['activation'],
                  kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(NN_CONFIG['dropout_rate']),
            
            # Hidden layers
            Dense(NN_CONFIG['hidden_layers'][1],
                  activation=NN_CONFIG['activation'],
                  kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(NN_CONFIG['dropout_rate']),
            
            Dense(NN_CONFIG['hidden_layers'][2],
                  activation=NN_CONFIG['activation'],
                  kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(NN_CONFIG['dropout_rate']),
            
            # Output layer
            Dense(num_classes, activation=NN_CONFIG['output_activation'])
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=NN_CONFIG['loss'],
            metrics=NN_CONFIG['metrics']
        )
        
        self.model = model
        print("Model architecture built successfully")
        print(model.summary())
        
        return model
    
    def create_callbacks(self):
        """Create training callbacks"""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=NN_CONFIG['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                'models/best_water_quality_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=0.0001,
                verbose=1
            )
        ]
        
        return callbacks
    
    def train(self, processed_data):
        """Train the neural network model"""
        print("=== Starting Model Training ===")
        
        # Extract training data
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        X_val = processed_data['X_val']
        y_val = processed_data['y_val']
        class_weights = processed_data['class_weights']
        
        # Build model if not already built
        if self.model is None:
            self.build_model(input_dim=X_train.shape[1])
        
        # Create callbacks
        callbacks = self.create_callbacks()
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=NN_CONFIG['epochs'],
            batch_size=NN_CONFIG['batch_size'],
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        print("=== Training Complete ===")
        
        return self.history
    
    def evaluate(self, processed_data):
        """Evaluate model performance"""
        print("=== Model Evaluation ===")
        
        X_test = processed_data['X_test']
        y_test = processed_data['y_test']
        y_test_raw = processed_data['y_test_raw']
        
        # Predictions
        y_pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test_raw, y_pred)
        print(f"Test Accuracy: {accuracy:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        report = classification_report(
            y_test_raw, y_pred,
            target_names=[QUALITY_LABELS[i] for i in range(4)],
            digits=4
        )
        print(report)
        
        # Confusion matrix
        cm = confusion_matrix(y_test_raw, y_pred)
        
        return {
            'accuracy': accuracy,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'classification_report': report,
            'confusion_matrix': cm
        }
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(self.history.history['loss'], label='Training Loss')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax2.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('models/training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, cm):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[QUALITY_LABELS[i] for i in range(4)],
                    yticklabels=[QUALITY_LABELS[i] for i in range(4)])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig('models/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, filepath='models/water_quality_model.h5'):
        """Save trained model"""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
        
        # Save preprocessor using its own save method
        preprocessor_path = filepath.replace('.h5', '_preprocessor.pkl')
        self.preprocessor.save_preprocessor(preprocessor_path)
    
    def load_model(self, filepath='models/water_quality_model.h5'):
        """Load saved model"""
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
        
        # Load preprocessor
        preprocessor_path = filepath.replace('.h5', '_preprocessor.pkl')
        if os.path.exists(preprocessor_path):
            self.preprocessor = WaterQualityPreprocessor.load_preprocessor(preprocessor_path)
        else:
            print(f"Preprocessor file {preprocessor_path} not found")
    
    def predict_single(self, tds, turbidity, ph):
        """Make prediction for a single water sample"""
        # Preprocess the sample
        sample_scaled = self.preprocessor.preprocess_single_sample(tds, turbidity, ph)
        
        # Make prediction
        pred_proba = self.model.predict(sample_scaled, verbose=0)
        pred_class = np.argmax(pred_proba, axis=1)[0]
        confidence = pred_proba[0][pred_class]
        
        return {
            'quality_class': pred_class,
            'quality_label': QUALITY_LABELS[pred_class],
            'confidence': confidence,
            'probabilities': {
                QUALITY_LABELS[i]: pred_proba[0][i] 
                for i in range(4)
            }
        }

def main():
    """Train and evaluate water quality model"""
    # Initialize model
    model = WaterQualityModel()
    
    # Load and preprocess data
    preprocessor = WaterQualityPreprocessor()
    df = preprocessor.load_data('data/water_quality_dataset.csv')
    
    if df is None:
        print("Please run data generation first:")
        print("python src/data_processing/generate_data.py")
        return
    
    # Preprocess data
    processed_data = preprocessor.preprocess_pipeline(df)
    
    # Train model
    history = model.train(processed_data)
    
    # Evaluate model
    results = model.evaluate(processed_data)
    
    # Plot results
    model.plot_training_history()
    model.plot_confusion_matrix(results['confusion_matrix'])
    
    # Save model
    model.save_model()
    
    # Test single prediction
    print("\n=== Testing Single Prediction ===")
    # Ensure the model has the fitted preprocessor
    model.preprocessor = preprocessor
    test_prediction = model.predict_single(tds=250, turbidity=0.8, ph=7.2)
    print(f"Sample: TDS=250, Turbidity=0.8, pH=7.2")
    print(f"Predicted Quality: {test_prediction['quality_label']}")
    print(f"Confidence: {test_prediction['confidence']:.4f}")
    
    return model, results

if __name__ == "__main__":
    main()
