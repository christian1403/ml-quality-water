"""
Main entry point for water quality prediction system
"""

import argparse
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_processing.generate_data import WaterQualityDataGenerator
from src.data_processing.preprocessor import WaterQualityPreprocessor
from src.models.train_model import WaterQualityModel
from src.models.predict import WaterQualityPredictor
from src.utils.analysis_utils import WaterQualityAnalyzer, get_water_quality_guidelines
from config.config import DATA_CONFIG

def generate_data():
    """Generate synthetic water quality dataset"""
    print("Generating water quality dataset...")
    generator = WaterQualityDataGenerator(
        n_samples=DATA_CONFIG['n_samples'],
        random_state=42
    )
    
    df = generator.save_dataset(DATA_CONFIG['output_file'])
    return df

def train_model():
    """Train the water quality prediction model"""
    print("Training water quality model...")
    
    # Initialize components
    model = WaterQualityModel()
    preprocessor = WaterQualityPreprocessor()
    
    # Load data
    df = preprocessor.load_data(DATA_CONFIG['output_file'])
    if df is None:
        print("Generating data first...")
        df = generate_data()
    
    # Preprocess data
    processed_data = preprocessor.preprocess_pipeline(df)
    
    # Set the model's preprocessor to the fitted one
    model.preprocessor = preprocessor
    
    # Train model
    history = model.train(processed_data)
    
    # Evaluate model
    results = model.evaluate(processed_data)
    
    # Plot results
    model.plot_training_history()
    model.plot_confusion_matrix(results['confusion_matrix'])
    
    # Save model
    model.save_model()
    
    return model, results

def predict_water_quality(tds, turbidity, ph):
    """Make a single prediction"""
    predictor = WaterQualityPredictor()
    
    if predictor.model is None:
        print("Model not found. Training model first...")
        train_model()
        predictor = WaterQualityPredictor()
    
    result = predictor.predict(tds, turbidity, ph)
    
    if "error" not in result:
        print(predictor.generate_report(tds, turbidity, ph))
    else:
        print(f"Error: {result['error']}")
    
    return result

def interactive_mode():
    """Run interactive prediction mode"""
    from src.models.predict import main as predict_main
    predict_main()

def analyze_data():
    """Analyze existing dataset"""
    from src.utils.analysis_utils import plot_data_distribution, plot_correlation_matrix, plot_feature_relationships
    
    # Load data
    try:
        df = pd.read_csv(DATA_CONFIG['output_file'])
        print(f"Loaded dataset: {df.shape}")
        
        # Generate analysis plots
        plot_data_distribution(df)
        plot_correlation_matrix(df)
        plot_feature_relationships(df)
        
        # Print summary statistics
        print("\n=== Dataset Summary ===")
        print(df.describe())
        
        print("\n=== Quality Distribution ===")
        print(df['quality_label'].value_counts())
        
    except FileNotFoundError:
        print("Dataset not found. Generate data first with: python main.py --generate-data")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Water Quality Prediction System')
    parser.add_argument('--generate-data', action='store_true', 
                       help='Generate synthetic water quality dataset')
    parser.add_argument('--train', action='store_true',
                       help='Train the neural network model')
    parser.add_argument('--predict', nargs=3, type=float, metavar=('TDS', 'TURBIDITY', 'PH'),
                       help='Predict water quality for given sensor readings')
    parser.add_argument('--interactive', action='store_true',
                       help='Run interactive prediction mode')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze dataset and generate plots')
    parser.add_argument('--guidelines', action='store_true',
                       help='Show water quality guidelines')
    
    args = parser.parse_args()
    
    if args.generate_data:
        generate_data()
    
    elif args.train:
        train_model()
    
    elif args.predict:
        tds, turbidity, ph = args.predict
        predict_water_quality(tds, turbidity, ph)
    
    elif args.interactive:
        interactive_mode()
    
    elif args.analyze:
        analyze_data()
    
    elif args.guidelines:
        print(get_water_quality_guidelines())
    
    else:
        print("Water Quality Prediction System")
        print("Available commands:")
        print("  --generate-data    Generate synthetic dataset")
        print("  --train           Train the ML model")
        print("  --predict TDS TURBIDITY PH    Make a prediction")
        print("  --interactive     Interactive prediction mode")
        print("  --analyze         Analyze dataset")
        print("  --guidelines      Show water quality guidelines")
        print("\nExample usage:")
        print("  python main.py --generate-data")
        print("  python main.py --train")
        print("  python main.py --predict 300 1.5 7.2")
        print("  python main.py --interactive")

if __name__ == "__main__":
    main()
