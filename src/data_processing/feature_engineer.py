"""
Advanced Water Quality Feature Engineering
Implements internationally recognized water quality indices and composite features
"""

import numpy as np
import pandas as pd
import math
from typing import Dict, List, Tuple

class WaterQualityFeatureEngineer:
    """
    Advanced feature engineering for water quality prediction
    Implements multiple water quality indices used globally:
    1. WHO Water Quality Index (WQI)
    2. Canadian Council of Ministers WQI (CCME-WQI)
    3. Oregon Water Quality Index (OWQI)
    4. Comprehensive Pollution Index (CPI)
    5. Water Quality Rating (WQR)
    """
    
    def __init__(self):
        self.feature_names = []
        self.standards = self._load_international_standards()
        
    def _load_international_standards(self):
        """Load international water quality standards"""
        return {
            'who': {
                'ph': {'excellent': (7.0, 7.5), 'good': (6.5, 8.5), 'acceptable': (6.0, 9.0)},
                'tds': {'excellent': 300, 'good': 600, 'acceptable': 900},
                'turbidity': {'excellent': 1, 'good': 4, 'acceptable': 10}
            },
            'epa': {
                'ph': {'excellent': (6.8, 7.2), 'good': (6.5, 8.5), 'acceptable': (6.0, 9.0)},
                'tds': {'excellent': 500, 'good': 1000, 'acceptable': 1500},
                'turbidity': {'excellent': 1, 'good': 5, 'acceptable': 15}
            },
            'canada': {
                'ph': {'excellent': (7.0, 8.0), 'good': (6.5, 8.5), 'acceptable': (6.0, 9.0)},
                'tds': {'excellent': 200, 'good': 500, 'acceptable': 1000},
                'turbidity': {'excellent': 2, 'good': 8, 'acceptable': 20}
            }
        }
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer comprehensive water quality features
        
        Args:
            df: DataFrame with 'tds', 'turbidity', 'ph' columns
            
        Returns:
            DataFrame with engineered features
        """
        print("🔧 Engineering advanced water quality features...")
        
        # Make copy to avoid modifying original
        engineered_df = df.copy()
        
        # 1. WHO Water Quality Index
        engineered_df = self._add_who_wqi(engineered_df)
        
        # 2. CCME Water Quality Index
        engineered_df = self._add_ccme_wqi(engineered_df)
        
        # 3. Comprehensive Pollution Index
        engineered_df = self._add_cpi(engineered_df)
        
        # 4. Parameter Ratios and Interactions
        engineered_df = self._add_parameter_interactions(engineered_df)
        
        # 5. Risk Assessment Features
        engineered_df = self._add_risk_features(engineered_df)
        
        # 6. Statistical Features
        engineered_df = self._add_statistical_features(engineered_df)
        
        # 7. Composite Health Indicators
        engineered_df = self._add_health_indicators(engineered_df)
        
        print(f"✅ Generated {len(engineered_df.columns) - len(df.columns)} new features")
        self.feature_names = [col for col in engineered_df.columns if col not in df.columns]
        
        return engineered_df
    
    def _add_who_wqi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add WHO Water Quality Index"""
        print("   📊 Computing WHO Water Quality Index...")
        
        def ph_subindex(ph):
            if 7.0 <= ph <= 7.5:
                return 100
            elif 6.5 <= ph <= 8.5:
                return 100 - abs(ph - 7.25) * 20
            elif 6.0 <= ph <= 9.0:
                return 70 - abs(ph - 7.25) * 15
            else:
                return max(0, 30 - abs(ph - 7.25) * 10)
        
        def tds_subindex(tds):
            if tds <= 300:
                return 100
            elif tds <= 600:
                return 100 - (tds - 300) / 3
            elif tds <= 900:
                return 70 - (tds - 600) / 6
            else:
                return max(0, 20 - (tds - 900) / 20)
        
        def turbidity_subindex(turb):
            if turb <= 1:
                return 100
            elif turb <= 5:
                return 100 - (turb - 1) * 10
            elif turb <= 10:
                return 60 - (turb - 5) * 8
            else:
                return max(0, 20 - (turb - 10) * 2)
        
        # Calculate sub-indices
        df['who_ph_index'] = df['ph'].apply(ph_subindex)
        df['who_tds_index'] = df['tds'].apply(tds_subindex)
        df['who_turbidity_index'] = df['turbidity'].apply(turbidity_subindex)
        
        # Overall WHO WQI (geometric mean for WHO methodology)
        df['who_wqi'] = np.power(
            df['who_ph_index'] * df['who_tds_index'] * df['who_turbidity_index'],
            1/3
        )
        
        return df
    
    def _add_ccme_wqi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Canadian Council of Ministers WQI"""
        print("   🍁 Computing CCME Water Quality Index...")
        
        # CCME methodology uses exceedance frequency and amplitude
        def ccme_exceedance(ph, tds, turbidity):
            failed_tests = 0
            total_tests = 3
            
            # Count exceedances
            if ph < 6.5 or ph > 8.5:
                failed_tests += 1
            if tds > 500:
                failed_tests += 1
            if turbidity > 5:
                failed_tests += 1
            
            f1 = (failed_tests / total_tests) * 100  # Scope
            f2 = (failed_tests / total_tests) * 100  # Frequency
            
            # Amplitude calculation (normalized exceedance)
            exceedances = []
            if ph < 6.5:
                exceedances.append((6.5 - ph) / 6.5)
            elif ph > 8.5:
                exceedances.append((ph - 8.5) / 8.5)
            
            if tds > 500:
                exceedances.append((tds - 500) / 500)
                
            if turbidity > 5:
                exceedances.append((turbidity - 5) / 5)
            
            if exceedances:
                nse = sum(exceedances) / len(exceedances)
                f3 = nse / (0.01 * nse + 0.01) * 100
            else:
                f3 = 0
            
            # CCME WQI calculation
            ccme_wqi = 100 - (math.sqrt(f1**2 + f2**2 + f3**2) / 1.732)
            return max(0, min(100, ccme_wqi))
        
        df['ccme_wqi'] = df.apply(
            lambda row: ccme_exceedance(row['ph'], row['tds'], row['turbidity']), 
            axis=1
        )
        
        return df
    
    def _add_cpi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Comprehensive Pollution Index"""
        print("   🏭 Computing Comprehensive Pollution Index...")
        
        # CPI methodology - higher values indicate more pollution
        def calculate_cpi(ph, tds, turbidity):
            # Normalize parameters to pollution scale (0-1)
            ph_pollution = 0
            if ph < 6.5:
                ph_pollution = (6.5 - ph) / 2.5  # Max at pH 4
            elif ph > 8.5:
                ph_pollution = (ph - 8.5) / 3.5   # Max at pH 12
            
            tds_pollution = min(1.0, tds / 2000)  # Max at 2000 mg/L
            turbidity_pollution = min(1.0, turbidity / 50)  # Max at 50 NTU
            
            # Weighted CPI (TDS has highest weight for drinking water)
            cpi = (0.3 * ph_pollution + 0.5 * tds_pollution + 0.2 * turbidity_pollution) * 100
            return cpi
        
        df['cpi'] = df.apply(
            lambda row: calculate_cpi(row['ph'], row['tds'], row['turbidity']), 
            axis=1
        )
        
        return df
    
    def _add_parameter_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add parameter interaction features"""
        print("   ⚗️ Computing parameter interactions...")
        
        # pH-TDS interactions
        df['ph_tds_product'] = df['ph'] * np.log1p(df['tds'])
        df['ph_tds_ratio'] = df['ph'] / (df['tds'] / 100 + 1)
        
        # TDS-Turbidity interactions  
        df['tds_turbidity_product'] = np.log1p(df['tds']) * np.log1p(df['turbidity'])
        df['tds_turbidity_ratio'] = df['tds'] / (df['turbidity'] + 1)
        
        # pH-Turbidity interactions
        df['ph_turbidity_product'] = df['ph'] * np.log1p(df['turbidity'])
        df['ph_turbidity_diff'] = abs(df['ph'] - 7) * df['turbidity']
        
        # Three-way interaction
        df['ph_tds_turbidity_index'] = (
            df['ph'] * np.log1p(df['tds']) * np.log1p(df['turbidity'])
        )
        
        return df
    
    def _add_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add risk assessment features"""
        print("   ⚠️ Computing risk assessment features...")
        
        # Individual parameter risks (0-1 scale)
        df['ph_risk'] = df['ph'].apply(lambda x: 
            max(0, (abs(x - 7.25) - 1.25) / 2.75) if abs(x - 7.25) > 1.25 else 0
        )
        
        df['tds_risk'] = df['tds'].apply(lambda x: 
            min(1.0, max(0, (x - 300) / 1700))
        )
        
        df['turbidity_risk'] = df['turbidity'].apply(lambda x: 
            min(1.0, max(0, (x - 1) / 49))
        )
        
        # Composite risk scores
        df['total_risk_score'] = df['ph_risk'] + df['tds_risk'] + df['turbidity_risk']
        df['max_risk_score'] = df[['ph_risk', 'tds_risk', 'turbidity_risk']].max(axis=1)
        df['risk_variance'] = df[['ph_risk', 'tds_risk', 'turbidity_risk']].var(axis=1)
        
        # Risk categories
        df['high_risk_count'] = (df[['ph_risk', 'tds_risk', 'turbidity_risk']] > 0.5).sum(axis=1)
        
        return df
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features"""
        print("   📈 Computing statistical features...")
        
        # Normalize original parameters for statistical analysis
        params = ['ph', 'tds', 'turbidity']
        
        # Z-scores based on typical ranges
        typical_means = {'ph': 7.0, 'tds': 500, 'turbidity': 5}
        typical_stds = {'ph': 1.0, 'tds': 400, 'turbidity': 10}
        
        for param in params:
            df[f'{param}_zscore'] = (df[param] - typical_means[param]) / typical_stds[param]
        
        # Parameter deviations from ideal
        df['ph_deviation'] = abs(df['ph'] - 7.0)
        df['tds_deviation'] = abs(df['tds'] - 150)  # Ideal TDS ~150
        df['turbidity_deviation'] = df['turbidity']  # Ideal turbidity = 0
        
        # Statistical measures across parameters (after normalization)
        normalized_params = [f'{param}_zscore' for param in params]
        df['param_mean'] = df[normalized_params].mean(axis=1)
        df['param_std'] = df[normalized_params].std(axis=1)
        df['param_range'] = df[normalized_params].max(axis=1) - df[normalized_params].min(axis=1)
        
        return df
    
    def _add_health_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add health-based indicators"""
        print("   🏥 Computing health indicators...")
        
        # Drinking water safety score (0-100)
        def drinking_safety_score(ph, tds, turbidity):
            score = 100
            
            # pH safety deductions
            if ph < 6.5 or ph > 8.5:
                score -= 30
            elif ph < 6.8 or ph > 8.2:
                score -= 15
            
            # TDS safety deductions  
            if tds > 1000:
                score -= 40
            elif tds > 500:
                score -= 20
            elif tds < 50:
                score -= 10  # Too low can be problematic too
            
            # Turbidity safety deductions
            if turbidity > 10:
                score -= 30
            elif turbidity > 4:
                score -= 15
            
            return max(0, score)
        
        df['drinking_safety_score'] = df.apply(
            lambda row: drinking_safety_score(row['ph'], row['tds'], row['turbidity']), 
            axis=1
        )
        
        # Aesthetic quality score (taste, odor, appearance)
        def aesthetic_score(ph, tds, turbidity):
            score = 100
            
            # pH affects taste
            if abs(ph - 7.0) > 0.5:
                score -= abs(ph - 7.0) * 20
            
            # TDS affects taste and mineral content
            if tds > 300:
                score -= (tds - 300) / 20
            elif tds < 100:
                score -= (100 - tds) / 10
            
            # Turbidity affects appearance
            score -= turbidity * 10
            
            return max(0, min(100, score))
        
        df['aesthetic_score'] = df.apply(
            lambda row: aesthetic_score(row['ph'], row['tds'], row['turbidity']), 
            axis=1
        )
        
        # Treatment difficulty index (higher = more treatment needed)
        def treatment_difficulty(ph, tds, turbidity):
            difficulty = 0
            
            # pH treatment difficulty
            difficulty += abs(ph - 7.0) * 10
            
            # TDS treatment difficulty (desalination is expensive)
            if tds > 1000:
                difficulty += (tds - 1000) / 50
            
            # Turbidity treatment difficulty
            difficulty += turbidity * 2
            
            return min(100, difficulty)
        
        df['treatment_difficulty'] = df.apply(
            lambda row: treatment_difficulty(row['ph'], row['tds'], row['turbidity']), 
            axis=1
        )
        
        return df
    
    def get_feature_importance_mapping(self) -> Dict[str, str]:
        """Get mapping of feature names to their descriptions"""
        return {
            'who_wqi': 'WHO Water Quality Index (0-100)',
            'ccme_wqi': 'Canadian Water Quality Index (0-100)', 
            'cpi': 'Comprehensive Pollution Index (0-100)',
            'drinking_safety_score': 'Drinking Water Safety Score (0-100)',
            'aesthetic_score': 'Water Aesthetic Quality Score (0-100)',
            'treatment_difficulty': 'Water Treatment Difficulty Index (0-100)',
            'total_risk_score': 'Total Parameter Risk Score',
            'ph_tds_product': 'pH-TDS Interaction Feature',
            'tds_turbidity_ratio': 'TDS-Turbidity Ratio Feature',
            'param_std': 'Parameter Variability Measure'
        }

# Global instance
feature_engineer = WaterQualityFeatureEngineer()
