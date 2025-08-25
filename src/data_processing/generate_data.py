"""
Data generation module for water quality dataset
Creates synthetic but realistic water quality data based on sensor readings
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import WATER_STANDARDS, QUALITY_LABELS, DATA_CONFIG

class WaterQualityDataGenerator:
    """Generate synthetic water quality data based on realistic sensor parameters"""
    
    def __init__(self, n_samples=10000, random_state=42):
        self.n_samples = n_samples
        self.random_state = random_state
        np.random.seed(random_state)
    
    def _determine_quality_score(self, tds, turbidity, ph):
        """
        Determine water quality based on sensor readings
        Returns quality score: 0=Poor, 1=Acceptable, 2=Good, 3=Excellent
        """
        ph_score = self._get_ph_score(ph)
        tds_score = self._get_tds_score(tds)
        turbidity_score = self._get_turbidity_score(turbidity)
        
        # Calculate weighted average (pH is most critical)
        weights = [0.4, 0.3, 0.3]  # pH, TDS, Turbidity
        scores = [ph_score, tds_score, turbidity_score]
        weighted_score = np.average(scores, weights=weights)
        
        # Convert to discrete quality levels
        if weighted_score >= 2.5:
            return 3  # Excellent
        elif weighted_score >= 1.5:
            return 2  # Good
        elif weighted_score >= 0.5:
            return 1  # Acceptable
        else:
            return 0  # Poor
    
    def _get_ph_score(self, ph):
        """Score pH value: 0-3 scale"""
        if 7.0 <= ph <= 7.5:
            return 3  # Excellent
        elif 6.5 <= ph <= 8.5:
            return 2  # Good
        elif 6.0 <= ph <= 9.0:
            return 1  # Acceptable
        else:
            return 0  # Poor
    
    def _get_tds_score(self, tds):
        """Score TDS value: 0-3 scale"""
        if tds <= 300:
            return 3  # Excellent
        elif tds <= 600:
            return 2  # Good
        elif tds <= 900:
            return 1  # Acceptable
        else:
            return 0  # Poor
    
    def _get_turbidity_score(self, turbidity):
        """Score turbidity value: 0-3 scale"""
        if turbidity <= 1:
            return 3  # Excellent
        elif turbidity <= 4:
            return 2  # Good
        elif turbidity <= 10:
            return 1  # Acceptable
        else:
            return 0  # Poor
    
    def generate_realistic_sample(self, quality_target=None):
        """Generate a single realistic water sample"""
        if quality_target is None:
            quality_target = np.random.choice([0, 1, 2, 3], p=[0.15, 0.25, 0.35, 0.25])
        
        if quality_target == 3:  # Excellent
            ph = np.random.normal(7.25, 0.15)
            tds = np.random.normal(200, 50)
            turbidity = np.random.exponential(0.5)
        elif quality_target == 2:  # Good
            ph = np.random.normal(7.0, 0.4)
            tds = np.random.normal(450, 100)
            turbidity = np.random.exponential(2.0)
        elif quality_target == 1:  # Acceptable
            ph = np.random.normal(6.8, 0.8)
            tds = np.random.normal(750, 150)
            turbidity = np.random.exponential(6.0)
        else:  # Poor
            ph = np.random.choice([
                np.random.normal(5.5, 0.5),  # Too acidic
                np.random.normal(9.5, 0.5)   # Too alkaline
            ])
            tds = np.random.normal(1200, 300)
            turbidity = np.random.exponential(15.0)
        
        # Apply realistic bounds
        ph = np.clip(ph, 4.0, 12.0)
        tds = np.clip(tds, 50, 3000)
        turbidity = np.clip(turbidity, 0.1, 50)
        
        return tds, turbidity, ph
    
    def generate_dataset(self):
        """Generate complete dataset"""
        data = []
        
        # Generate samples for each quality level
        quality_distribution = [0.15, 0.25, 0.35, 0.25]  # Poor, Acceptable, Good, Excellent
        
        for quality in range(4):
            n_samples_quality = int(self.n_samples * quality_distribution[quality])
            
            for _ in range(n_samples_quality):
                tds, turbidity, ph = self.generate_realistic_sample(quality)
                
                # Verify quality matches expectations (with some noise)
                actual_quality = self._determine_quality_score(tds, turbidity, ph)
                
                # Add some noise to make it more realistic
                if np.random.random() < 0.1:  # 10% noise
                    actual_quality = np.random.choice([max(0, actual_quality-1), 
                                                     min(3, actual_quality+1)])
                
                data.append({
                    'tds': round(tds, 2),
                    'turbidity': round(turbidity, 2),
                    'ph': round(ph, 2),
                    'quality': actual_quality,
                    'quality_label': QUALITY_LABELS[actual_quality]
                })
        
        # Shuffle the dataset
        np.random.shuffle(data)
        
        return pd.DataFrame(data)
    
    def save_dataset(self, filepath):
        """Generate and save dataset to CSV"""
        df = self.generate_dataset()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        df.to_csv(filepath, index=False)
        print(f"Dataset saved to {filepath}")
        print(f"Dataset shape: {df.shape}")
        print(f"Quality distribution:\n{df['quality_label'].value_counts()}")
        
        return df

def main():
    """Generate water quality dataset"""
    generator = WaterQualityDataGenerator(
        n_samples=DATA_CONFIG['n_samples'],
        random_state=42
    )
    
    df = generator.save_dataset(DATA_CONFIG['output_file'])
    
    # Display basic statistics
    print("\n=== Dataset Statistics ===")
    print(df.describe())
    
    print("\n=== Feature Ranges by Quality ===")
    for quality in df['quality_label'].unique():
        subset = df[df['quality_label'] == quality]
        print(f"\n{quality}:")
        print(f"  pH: {subset['ph'].min():.2f} - {subset['ph'].max():.2f}")
        print(f"  TDS: {subset['tds'].min():.2f} - {subset['tds'].max():.2f}")
        print(f"  Turbidity: {subset['turbidity'].min():.2f} - {subset['turbidity'].max():.2f}")

if __name__ == "__main__":
    main()
