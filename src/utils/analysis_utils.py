"""
Utility functions for water quality analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import QUALITY_LABELS, WATER_STANDARDS

def plot_data_distribution(df, save_path='models/data_distribution.png'):
    """Plot distribution of features by water quality"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    features = ['tds', 'turbidity', 'ph']
    
    # Feature distributions
    for i, feature in enumerate(features):
        ax = axes[i//2, i%2]
        
        for quality in df['quality_label'].unique():
            subset = df[df['quality_label'] == quality]
            ax.hist(subset[feature], alpha=0.7, label=quality, bins=30)
        
        ax.set_xlabel(feature.upper())
        ax.set_ylabel('Frequency')
        ax.set_title(f'{feature.upper()} Distribution by Quality')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Quality distribution
    ax = axes[1, 1]
    quality_counts = df['quality_label'].value_counts()
    ax.pie(quality_counts.values, labels=quality_counts.index, autopct='%1.1f%%')
    ax.set_title('Overall Quality Distribution')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_correlation_matrix(df, save_path='models/correlation_matrix.png'):
    """Plot correlation matrix of features"""
    # Select only numeric columns
    numeric_df = df[['tds', 'turbidity', 'ph', 'quality']].copy()
    
    correlation_matrix = numeric_df.corr()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.3f')
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_feature_relationships(df, save_path='models/feature_relationships.png'):
    """Plot pairwise relationships between features"""
    # Create pairplot
    plt.figure(figsize=(12, 10))
    
    # Select relevant columns
    plot_df = df[['tds', 'turbidity', 'ph', 'quality_label']].copy()
    
    # Create pairplot
    g = sns.pairplot(plot_df, hue='quality_label', diag_kind='hist',
                     plot_kws={'alpha': 0.6}, diag_kws={'alpha': 0.7})
    
    g.fig.suptitle('Feature Relationships by Water Quality', y=1.02)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def analyze_feature_importance(model, feature_names):
    """Analyze feature importance using permutation"""
    # This is a simplified version - for more detailed analysis,
    # you might want to use SHAP or other interpretation methods
    
    print("=== Feature Analysis ===")
    print("Feature names:", feature_names)
    print("For detailed feature importance analysis, consider using:")
    print("- SHAP (SHapley Additive exPlanations)")
    print("- LIME (Local Interpretable Model-agnostic Explanations)")
    print("- Permutation importance")

def validate_sensor_reading(tds=None, turbidity=None, ph=None):
    """Validate sensor readings against realistic ranges"""
    errors = []
    warnings = []
    
    # pH validation
    if ph is not None:
        if ph < 4.0 or ph > 12.0:
            errors.append(f"pH {ph} is outside realistic range (4.0-12.0)")
        elif ph < 6.0 or ph > 9.0:
            warnings.append(f"pH {ph} is outside typical drinking water range (6.0-9.0)")
    
    # TDS validation
    if tds is not None:
        if tds < 0 or tds > 5000:
            errors.append(f"TDS {tds} mg/L is outside realistic range (0-5000)")
        elif tds > 1000:
            warnings.append(f"TDS {tds} mg/L is high for drinking water")
    
    # Turbidity validation
    if turbidity is not None:
        if turbidity < 0 or turbidity > 100:
            errors.append(f"Turbidity {turbidity} NTU is outside realistic range (0-100)")
        elif turbidity > 10:
            warnings.append(f"Turbidity {turbidity} NTU is high for drinking water")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings
    }

def get_water_quality_guidelines():
    """Return WHO/EPA water quality guidelines"""
    guidelines = """
    === WATER QUALITY GUIDELINES ===
    
    pH (Potential of Hydrogen):
    - Optimal: 7.0 - 7.5
    - Acceptable: 6.5 - 8.5 (WHO/EPA standard)
    - Outside range: Requires treatment
    
    TDS (Total Dissolved Solids):
    - Excellent: < 300 mg/L
    - Good: 300 - 600 mg/L
    - Acceptable: 600 - 900 mg/L
    - Poor: > 900 mg/L
    
    Turbidity:
    - Excellent: < 1 NTU
    - Good: 1 - 4 NTU
    - Acceptable: 4 - 10 NTU
    - Poor: > 10 NTU
    
    Overall Quality Classification:
    - Excellent: All parameters in optimal range
    - Good: Most parameters in good range
    - Acceptable: Parameters within safe limits
    - Poor: One or more parameters require attention
    """
    
    return guidelines

def calculate_water_quality_index(tds, turbidity, ph):
    """Calculate a simple Water Quality Index (WQI)"""
    # Normalize each parameter (0-100 scale)
    
    # pH normalization (7.0 is optimal)
    ph_optimal = 7.0
    ph_normalized = max(0, 100 - abs(ph - ph_optimal) * 20)
    
    # TDS normalization (lower is better)
    tds_normalized = max(0, 100 - (tds / 10))
    
    # Turbidity normalization (lower is better)
    turbidity_normalized = max(0, 100 - (turbidity * 10))
    
    # Weighted average
    weights = [0.4, 0.3, 0.3]  # pH, TDS, Turbidity
    wqi = np.average([ph_normalized, tds_normalized, turbidity_normalized], weights=weights)
    
    return min(100, max(0, wqi))

def format_prediction_output(prediction_result):
    """Format prediction results for display"""
    if "error" in prediction_result:
        return f"❌ Error: {prediction_result['error']}"
    
    result = prediction_result['prediction']
    quality_emoji = {
        'Excellent': '🟢',
        'Good': '🔵', 
        'Acceptable': '🟡',
        'Poor': '🔴'
    }
    
    emoji = quality_emoji.get(result['quality_label'], '⚪')
    
    output = f"""
{emoji} Water Quality: {result['quality_label']}
📊 Confidence: {result['confidence']:.1%}
💧 Recommendation: {prediction_result['recommendation']}

Sensor Readings:
- TDS: {prediction_result['input']['tds']} mg/L
- Turbidity: {prediction_result['input']['turbidity']} NTU  
- pH: {prediction_result['input']['ph']}
"""
    
    return output

class WaterQualityAnalyzer:
    """Comprehensive water quality analysis tool"""
    
    def __init__(self):
        self.predictor = None
    
    def load_predictor(self):
        """Load the trained predictor"""
        from models.predict import WaterQualityPredictor
        self.predictor = WaterQualityPredictor()
    
    def comprehensive_analysis(self, tds, turbidity, ph):
        """Perform comprehensive water quality analysis"""
        if self.predictor is None:
            self.load_predictor()
        
        # Validate inputs
        validation = validate_sensor_reading(tds, turbidity, ph)
        
        if not validation['valid']:
            return {"error": "; ".join(validation['errors'])}
        
        # Get ML prediction
        ml_prediction = self.predictor.predict(tds, turbidity, ph)
        
        # Calculate WQI
        wqi = calculate_water_quality_index(tds, turbidity, ph)
        
        # Combine results
        result = {
            'ml_prediction': ml_prediction,
            'water_quality_index': wqi,
            'validation_warnings': validation['warnings'],
            'parameter_analysis': self._detailed_parameter_analysis(tds, turbidity, ph)
        }
        
        return result
    
    def _detailed_parameter_analysis(self, tds, turbidity, ph):
        """Detailed analysis of each parameter"""
        analysis = {}
        
        # pH analysis
        if ph < 6.5:
            analysis['ph'] = f"Acidic ({ph}) - may cause corrosion"
        elif ph > 8.5:
            analysis['ph'] = f"Alkaline ({ph}) - may cause scaling"
        else:
            analysis['ph'] = f"Within acceptable range ({ph})"
        
        # TDS analysis
        if tds > 1000:
            analysis['tds'] = f"High TDS ({tds} mg/L) - may affect taste"
        elif tds < 50:
            analysis['tds'] = f"Very low TDS ({tds} mg/L) - may lack minerals"
        else:
            analysis['tds'] = f"Acceptable TDS level ({tds} mg/L)"
        
        # Turbidity analysis
        if turbidity > 4:
            analysis['turbidity'] = f"High turbidity ({turbidity} NTU) - filtration recommended"
        else:
            analysis['turbidity'] = f"Acceptable clarity ({turbidity} NTU)"
        
        return analysis
