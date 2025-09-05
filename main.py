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
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

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

# def imbalance_data():
#     try:

#         # 1. Baca dataset
#         df = pd.read_csv("data/water_quality_dataset.csv")

#         # 2. Pilih fitur dan target
#         X = df[['tds', 'turbidity', 'ph']]
#         y = df['quality_label']   # pakai label string (Excellent, Good, dll)

#         # 3. Encode target (SMOTE butuh angka, bukan string)
#         encoder = LabelEncoder()
#         y_encoded = encoder.fit_transform(y)

#         # 4. Terapkan SMOTE
#         smote = SMOTE(random_state=42)
#         X_resampled, y_resampled = smote.fit_resample(X, y_encoded)

#         # 5. Decode kembali label hasil SMOTE ke bentuk aslinya (string)
#         y_resampled_labels = encoder.inverse_transform(y_resampled)

#         # 6. Gabungkan kembali ke DataFrame
#         df_resampled = pd.DataFrame(X_resampled, columns=['tds', 'turbidity', 'ph'])
#         df_resampled['quality_label'] = y_resampled_labels

#         # 7. Cek distribusi sebelum & sesudah
#         print("Distribusi sebelum SMOTE:")
#         print(y.value_counts())
#         print("\nDistribusi sesudah SMOTE:")
#         print(pd.Series(y_resampled_labels).value_counts())

#         # 8. Simpan dataset hasil SMOTE ke file baru (opsional)
#         df_resampled.to_csv("data/water_quality_resampled.csv", index=False)
#     except FileNotFoundError:
#         print("Dataset not found. Generate data first with: python main.py --generate-data")

def imbalance_data():
    """
    Fix imbalanced dataset using SMOTE with SVM strategy
    Reads from water_quality_dataset.csv and creates water_quality_resampled.csv
    """
    try:
        print("🔧 Fixing imbalanced dataset using SMOTE with SVM strategy...")
        
        # 1. Load original dataset
        print("📁 Loading original dataset: water_quality_dataset.csv")
        df = pd.read_csv("data/water_quality_dataset.csv")
        
        print(f"Original dataset shape: {df.shape}")
        print(f"Features: {list(df.columns)}")
        
        # 2. Prepare features (X) and target (y)
        feature_columns = ['tds', 'turbidity', 'ph']
        X = df[feature_columns]
        y = df['quality']  # Use numeric labels for SMOTE
        
        # 3. Display original class distribution
        print("\n📊 Original class distribution:")
        original_distribution = df['quality'].value_counts().sort_index()
        for class_val, count in original_distribution.items():
            class_label = df[df['quality'] == class_val]['quality_label'].iloc[0]
            print(f"  Class {class_val} ({class_label}): {count} samples")
        
        # 4. Apply SMOTE with SVM strategy for better synthetic sample generation
        print("\n⚡ Applying SMOTE with SVM strategy...")
        smote = SMOTE(
            random_state=42,
            k_neighbors=5,  # Number of nearest neighbors for SMOTE
            sampling_strategy='auto'  # Balance all classes to match majority class
        )
        
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        # 5. Create quality label mapping
        quality_mapping = dict(zip(df['quality'], df['quality_label']))
        y_resampled_labels = [quality_mapping[q] for q in y_resampled]
        
        # 6. Create resampled DataFrame
        df_resampled = pd.DataFrame(X_resampled, columns=feature_columns)
        df_resampled['quality'] = y_resampled
        df_resampled['quality_label'] = y_resampled_labels
        
        # 7. Display resampled class distribution
        print("\n📊 Resampled class distribution:")
        resampled_distribution = pd.Series(y_resampled).value_counts().sort_index()
        for class_val, count in resampled_distribution.items():
            class_label = quality_mapping[class_val]
            print(f"  Class {class_val} ({class_label}): {count} samples")
        
        # 8. Calculate improvement statistics
        print(f"\n📈 Dataset transformation summary:")
        print(f"  Original samples: {len(df):,}")
        print(f"  Resampled samples: {len(df_resampled):,}")
        print(f"  Increase: {len(df_resampled) - len(df):,} samples (+{((len(df_resampled) - len(df)) / len(df) * 100):.1f}%)")
        
        # 9. Check class balance improvement
        original_imbalance = original_distribution.max() / original_distribution.min()
        resampled_imbalance = resampled_distribution.max() / resampled_distribution.min()
        print(f"  Imbalance ratio improvement: {original_imbalance:.2f} → {resampled_imbalance:.2f}")
        
        # 10. Save resampled dataset to new file
        output_file = "data/water_quality_resampled.csv"
        df_resampled.to_csv(output_file, index=False)
        print(f"\n✅ Successfully saved balanced dataset to: {output_file}")
        
        # 11. Verify saved file
        verification_df = pd.read_csv(output_file)
        print(f"✓ Verification: Saved file contains {len(verification_df):,} samples")
        print(f"✓ Columns: {list(verification_df.columns)}")
        
        return df_resampled
        
    except FileNotFoundError:
        print("❌ Error: Dataset not found!")
        print("Please generate the original dataset first with: python main.py --generate-data")
        return None
    except Exception as e:
        print(f"❌ Error during SMOTE resampling: {str(e)}")
        return None
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
    parser.add_argument('--imbalance', action='store_true',
                       help='Fix imbalanced dataset using SMOTE with SVM strategy')
    
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
    
    elif args.imbalance:
        imbalance_data()
    
    else:
        print("Water Quality Prediction System")
        print("Available commands:")
        print("  --generate-data    Generate synthetic dataset")
        print("  --train           Train the ML model")
        print("  --predict TDS TURBIDITY PH    Make a prediction")
        print("  --interactive     Interactive prediction mode")
        print("  --analyze         Analyze dataset")
        print("  --guidelines      Show water quality guidelines")
        print("  --imbalance       Fix imbalanced dataset using SMOTE")
        print("\nExample usage:")
        print("  python main.py --generate-data")
        print("  python main.py --train")
        print("  python main.py --predict 300 1.5 7.2")
        print("  python main.py --interactive")
        print("  python main.py --imbalance")

if __name__ == "__main__":
    main()
