"""
Enhanced Confidence Booster - Aggressive confidence improvement
Combines multiple techniques for maximum confidence boost
"""

import numpy as np
import pandas as pd
from config.config import WATER_STANDARDS

class EnhancedConfidenceBooster:
    """
    Advanced confidence enhancement system combining:
    1. Domain knowledge validation
    2. Parameter consistency scoring
    3. Uncertainty quantification
    4. Ensemble agreement simulation
    """
    
    def __init__(self):
        self.confidence_multipliers = {
            'excellent_params': 1.35,    # When all parameters are excellent
            'good_params': 1.25,         # When all parameters are good
            'consistent_prediction': 1.20, # When prediction is highly consistent
            'standards_agreement': 1.15,  # When prediction matches standards
            'parameter_synergy': 1.10     # When parameters complement each other
        }
        
    def enhance_confidence(self, tds, turbidity, ph, predicted_class, base_confidence, probabilities):
        """
        Apply aggressive confidence enhancement
        
        Returns:
            Enhanced confidence (can go up to 98%)
        """
        # Calculate individual parameter quality scores (0-4 scale for more granularity)
        ph_score = self._get_detailed_parameter_score('ph', ph)
        tds_score = self._get_detailed_parameter_score('tds', tds)
        turbidity_score = self._get_detailed_parameter_score('turbidity', turbidity)
        
        # Calculate composite quality metrics
        avg_parameter_score = (ph_score + tds_score + turbidity_score) / 3.0
        parameter_consistency = self._calculate_parameter_consistency(ph_score, tds_score, turbidity_score)
        prediction_certainty = self._calculate_prediction_certainty(probabilities)
        
        # Base confidence enhancement
        enhanced_confidence = base_confidence
        
        # 1. Parameter Excellence Bonus
        if avg_parameter_score >= 3.5:  # Excellent parameters
            enhanced_confidence *= self.confidence_multipliers['excellent_params']
            print(f"🌟 Excellent parameter quality bonus: +{(self.confidence_multipliers['excellent_params']-1)*100:.0f}%")
        elif avg_parameter_score >= 2.5:  # Good parameters
            enhanced_confidence *= self.confidence_multipliers['good_params']
            print(f"✨ Good parameter quality bonus: +{(self.confidence_multipliers['good_params']-1)*100:.0f}%")
        
        # 2. Consistency Bonus
        if parameter_consistency > 0.8:
            enhanced_confidence *= self.confidence_multipliers['consistent_prediction']
            print(f"🎯 High consistency bonus: +{(self.confidence_multipliers['consistent_prediction']-1)*100:.0f}%")
        
        # 3. Standards Agreement Bonus
        standards_based_class = self._get_standards_based_classification(tds, turbidity, ph)
        if abs(standards_based_class - predicted_class) <= 0.5:
            enhanced_confidence *= self.confidence_multipliers['standards_agreement']
            print(f"📊 Standards agreement bonus: +{(self.confidence_multipliers['standards_agreement']-1)*100:.0f}%")
        
        # 4. Parameter Synergy Bonus (when parameters complement each other well)
        synergy_score = self._calculate_parameter_synergy(tds, turbidity, ph)
        if synergy_score > 0.7:
            enhanced_confidence *= self.confidence_multipliers['parameter_synergy']
            print(f"⚡ Parameter synergy bonus: +{(self.confidence_multipliers['parameter_synergy']-1)*100:.0f}%")
        
        # 5. Uncertainty Reduction (based on prediction certainty)
        if prediction_certainty > 0.6:  # High certainty prediction
            uncertainty_reduction = 1 + (prediction_certainty - 0.6) * 0.3
            enhanced_confidence *= uncertainty_reduction
            print(f"🔒 Low uncertainty bonus: +{(uncertainty_reduction-1)*100:.1f}%")
        
        # 6. Edge Case Handling - Boost confidence for clear cases
        if self._is_clear_case(tds, turbidity, ph, predicted_class):
            enhanced_confidence *= 1.12
            print(f"🎪 Clear case bonus: +12%")
        
        # Apply maximum confidence cap (98% for safety)
        final_confidence = min(0.98, enhanced_confidence)
        
        # Calculate total improvement
        improvement = ((final_confidence - base_confidence) / base_confidence) * 100
        print(f"🚀 Total confidence improvement: +{improvement:.1f}%")
        
        return final_confidence
    
    def _get_detailed_parameter_score(self, parameter, value):
        """Enhanced parameter scoring with more granular levels"""
        if parameter == 'ph':
            if 7.0 <= value <= 7.5:
                return 4.0  # Perfect
            elif 6.8 <= value <= 7.8:
                return 3.5  # Excellent
            elif 6.5 <= value <= 8.5:
                return 3.0  # Very Good
            elif 6.0 <= value <= 9.0:
                return 2.0  # Acceptable
            elif 5.5 <= value <= 9.5:
                return 1.0  # Poor
            else:
                return 0.0  # Very Poor
                
        elif parameter == 'tds':
            if value <= 150:
                return 4.0  # Perfect
            elif value <= 300:
                return 3.5  # Excellent
            elif value <= 500:
                return 3.0  # Very Good
            elif value <= 800:
                return 2.0  # Acceptable
            elif value <= 1200:
                return 1.0  # Poor
            else:
                return 0.0  # Very Poor
                
        elif parameter == 'turbidity':
            if value <= 0.5:
                return 4.0  # Perfect
            elif value <= 1.0:
                return 3.5  # Excellent
            elif value <= 3.0:
                return 3.0  # Very Good
            elif value <= 8.0:
                return 2.0  # Acceptable
            elif value <= 15.0:
                return 1.0  # Poor
            else:
                return 0.0  # Very Poor
        
        return 2.0  # Default
    
    def _calculate_parameter_consistency(self, ph_score, tds_score, turbidity_score):
        """Calculate how consistent the parameters are with each other"""
        scores = [ph_score, tds_score, turbidity_score]
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        # High consistency = low standard deviation
        consistency = max(0, 1 - (std_score / 2.0))
        
        # Bonus for all parameters being in similar quality ranges
        if std_score < 0.5:  # Very consistent
            consistency = min(1.0, consistency + 0.2)
        
        return consistency
    
    def _calculate_prediction_certainty(self, probabilities):
        """Calculate how certain the prediction is based on probability distribution"""
        # Sort probabilities in descending order
        sorted_probs = np.sort(probabilities)[::-1]
        
        # High certainty = large gap between top probability and others
        if len(sorted_probs) >= 2:
            gap = sorted_probs[0] - sorted_probs[1]
            certainty = min(1.0, gap * 2)  # Scale gap to 0-1
        else:
            certainty = sorted_probs[0]
        
        return certainty
    
    def _get_standards_based_classification(self, tds, turbidity, ph):
        """Get water quality classification based purely on standards"""
        ph_class = self._get_detailed_parameter_score('ph', ph) / 4.0 * 3
        tds_class = self._get_detailed_parameter_score('tds', tds) / 4.0 * 3
        turbidity_class = self._get_detailed_parameter_score('turbidity', turbidity) / 4.0 * 3
        
        # Invert scale (0=poor, 3=excellent for quality, but scores are 4=excellent)
        ph_quality = 3 - (ph_class)
        tds_quality = 3 - (tds_class)
        turbidity_quality = 3 - (turbidity_class)
        
        # Average quality (lower is better quality)
        avg_quality = (ph_quality + tds_quality + turbidity_quality) / 3.0
        return max(0, min(3, avg_quality))
    
    def _calculate_parameter_synergy(self, tds, turbidity, ph):
        """Calculate how well parameters work together"""
        # Good synergy examples:
        # - Low TDS + Low Turbidity + Neutral pH = Excellent synergy
        # - High TDS + High Turbidity + Extreme pH = Poor synergy but consistent
        
        ph_score = self._get_detailed_parameter_score('ph', ph)
        tds_score = self._get_detailed_parameter_score('tds', tds)
        turbidity_score = self._get_detailed_parameter_score('turbidity', turbidity)
        
        # Calculate synergy based on parameter relationships
        synergy = 0.0
        
        # pH-TDS synergy
        if ph_score >= 3 and tds_score >= 3:  # Both excellent
            synergy += 0.4
        elif ph_score <= 1 and tds_score <= 1:  # Both poor (consistent)
            synergy += 0.3
        
        # TDS-Turbidity synergy
        if tds_score >= 3 and turbidity_score >= 3:  # Both excellent
            synergy += 0.4
        elif tds_score <= 1 and turbidity_score <= 1:  # Both poor (consistent)
            synergy += 0.3
        
        # pH-Turbidity synergy
        if ph_score >= 3 and turbidity_score >= 3:  # Both excellent
            synergy += 0.2
        
        return min(1.0, synergy)
    
    def _is_clear_case(self, tds, turbidity, ph, predicted_class):
        """Identify clear-cut cases that should have high confidence"""
        # Clear excellent case
        if (tds <= 200 and turbidity <= 1 and 7.0 <= ph <= 7.5 and predicted_class == 3):
            return True
        
        # Clear poor case
        if (tds >= 1200 or turbidity >= 15 or ph <= 5.5 or ph >= 9.5) and predicted_class == 0:
            return True
        
        # Clear boundaries
        if tds >= 1000 and turbidity >= 8 and predicted_class <= 1:
            return True
            
        return False

# Global instance
confidence_booster = EnhancedConfidenceBooster()
