"""
Real-time water quality monitoring simulation
"""

import time
import random
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.predict import WaterQualityPredictor
from src.utils.analysis_utils import format_prediction_output

class WaterQualityMonitor:
    """Simulate real-time water quality monitoring"""
    
    def __init__(self):
        self.predictor = WaterQualityPredictor()
        self.monitoring_log = []
    
    def simulate_sensor_reading(self, base_quality='good'):
        """Simulate realistic sensor readings"""
        
        if base_quality == 'excellent':
            tds = np.random.normal(220, 30)
            turbidity = np.random.exponential(0.6)
            ph = np.random.normal(7.2, 0.1)
        elif base_quality == 'good':
            tds = np.random.normal(400, 80)
            turbidity = np.random.exponential(2.0)
            ph = np.random.normal(7.0, 0.3)
        elif base_quality == 'acceptable':
            tds = np.random.normal(750, 100)
            turbidity = np.random.exponential(5.0)
            ph = np.random.normal(6.9, 0.4)
        else:  # poor
            tds = np.random.normal(1300, 200)
            turbidity = np.random.exponential(12.0)
            ph = np.random.choice([
                np.random.normal(5.8, 0.2),
                np.random.normal(8.8, 0.2)
            ])
        
        # Apply realistic bounds and add sensor noise
        tds = max(50, min(3000, tds + np.random.normal(0, 5)))
        turbidity = max(0.1, min(50, turbidity + np.random.normal(0, 0.1)))
        ph = max(4.0, min(12.0, ph + np.random.normal(0, 0.05)))
        
        return round(tds, 1), round(turbidity, 1), round(ph, 1)
    
    def continuous_monitoring(self, duration_minutes=5, interval_seconds=30):
        """Run continuous monitoring simulation"""
        print("🔄 Starting continuous water quality monitoring...")
        print(f"⏱️ Duration: {duration_minutes} minutes, Interval: {interval_seconds} seconds")
        print("=" * 80)
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        monitoring_data = []
        
        try:
            while time.time() < end_time:
                # Simulate sensor reading
                quality_scenarios = ['excellent', 'good', 'acceptable', 'poor']
                base_quality = np.random.choice(quality_scenarios, p=[0.3, 0.4, 0.2, 0.1])
                
                tds, turbidity, ph = self.simulate_sensor_reading(base_quality)
                
                # Make prediction
                result = self.predictor.predict(tds, turbidity, ph)
                
                if "error" not in result:
                    # Display real-time result
                    timestamp = time.strftime("%H:%M:%S")
                    quality = result['prediction']['quality_label']
                    confidence = result['prediction']['confidence']
                    
                    # Color coding for terminal output
                    quality_colors = {
                        'Excellent': '\\033[92m',  # Green
                        'Good': '\\033[94m',       # Blue
                        'Acceptable': '\\033[93m', # Yellow
                        'Poor': '\\033[91m'       # Red
                    }
                    reset_color = '\\033[0m'
                    
                    color = quality_colors.get(quality, '')
                    
                    print(f"[{timestamp}] TDS:{tds:6.1f} | Turb:{turbidity:5.1f} | pH:{ph:4.1f} | "
                          f"{color}{quality:10s}{reset_color} ({confidence:.1%})")
                    
                    # Store monitoring data
                    monitoring_data.append({
                        'timestamp': timestamp,
                        'tds': tds,
                        'turbidity': turbidity,
                        'ph': ph,
                        'quality': quality,
                        'confidence': confidence
                    })
                
                # Wait for next reading
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            print("\\n⏹️ Monitoring stopped by user")
        
        # Generate monitoring report
        if monitoring_data:
            self.generate_monitoring_report(monitoring_data)
        
        return monitoring_data
    
    def generate_monitoring_report(self, monitoring_data):
        """Generate monitoring session report"""
        df = pd.DataFrame(monitoring_data)
        
        print("\\n" + "=" * 80)
        print("📊 MONITORING SESSION REPORT")
        print("=" * 80)
        
        print(f"📈 Total readings: {len(df)}")
        print(f"⏱️ Session duration: {df.iloc[-1]['timestamp']} - {df.iloc[0]['timestamp']}")
        
        # Quality distribution
        quality_dist = df['quality'].value_counts()
        print(f"\\n💧 Quality distribution:")
        for quality, count in quality_dist.items():
            percentage = count / len(df) * 100
            print(f"   {quality}: {count} readings ({percentage:.1f}%)")
        
        # Average parameters by quality
        print(f"\\n📊 Average parameters by quality:")
        avg_by_quality = df.groupby('quality')[['tds', 'turbidity', 'ph']].mean()
        print(avg_by_quality.round(2))
        
        # Alerts and warnings
        poor_readings = df[df['quality'] == 'Poor']
        if len(poor_readings) > 0:
            print(f"\\n⚠️ ALERTS: {len(poor_readings)} poor quality readings detected!")
            print("   Immediate attention required for water treatment.")
        
        low_confidence = df[df['confidence'] < 0.7]
        if len(low_confidence) > 0:
            print(f"\\n🔍 INFO: {len(low_confidence)} low confidence predictions.")
            print("   Consider additional sensor calibration.")
        
        print("\\n" + "=" * 80)

def main():
    """Main monitoring interface"""
    print("🌊 Water Quality Real-time Monitoring System")
    print("=" * 60)
    
    monitor = WaterQualityMonitor()
    
    if monitor.predictor.model is None:
        print("❌ Model not loaded. Please train the model first:")
        print("python src/models/train_model.py")
        return
    
    print("Available monitoring modes:")
    print("1. Single reading")
    print("2. Continuous monitoring")
    print("3. Batch simulation")
    print("4. Custom sensor input")
    
    try:
        choice = input("\\nSelect mode (1-4): ").strip()
        
        if choice == "1":
            # Single reading
            quality = input("Enter base quality (excellent/good/acceptable/poor) or press Enter for random: ").strip().lower()
            if quality not in ['excellent', 'good', 'acceptable', 'poor']:
                quality = np.random.choice(['excellent', 'good', 'acceptable', 'poor'])
            
            tds, turbidity, ph = monitor.simulate_sensor_reading(quality)
            result = monitor.predictor.predict(tds, turbidity, ph)
            
            print(f"\\n📊 Simulated reading:")
            print(f"TDS: {tds} mg/L")
            print(f"Turbidity: {turbidity} NTU")
            print(f"pH: {ph}")
            
            if "error" not in result:
                print(format_prediction_output(result))
            else:
                print(f"Error: {result['error']}")
        
        elif choice == "2":
            # Continuous monitoring
            duration = float(input("Enter monitoring duration in minutes (default 2): ") or "2")
            interval = float(input("Enter reading interval in seconds (default 10): ") or "10")
            
            monitor.continuous_monitoring(duration, interval)
        
        elif choice == "3":
            # Batch simulation
            n_samples = int(input("Enter number of samples to simulate (default 20): ") or "20")
            
            print(f"\\n🧪 Simulating {n_samples} water samples...")
            
            for i in range(n_samples):
                quality = np.random.choice(['excellent', 'good', 'acceptable', 'poor'], 
                                         p=[0.25, 0.35, 0.25, 0.15])
                tds, turbidity, ph = monitor.simulate_sensor_reading(quality)
                result = monitor.predictor.predict(tds, turbidity, ph)
                
                if "error" not in result:
                    predicted_quality = result['prediction']['quality_label']
                    confidence = result['prediction']['confidence']
                    print(f"Sample {i+1:2d}: TDS={tds:6.1f} | Turb={turbidity:5.1f} | pH={ph:4.1f} | "
                          f"Quality: {predicted_quality:10s} ({confidence:.1%})")
        
        elif choice == "4":
            # Custom input
            print("\\n🔧 Enter custom sensor readings:")
            tds = float(input("TDS (mg/L): "))
            turbidity = float(input("Turbidity (NTU): "))
            ph = float(input("pH: "))
            
            result = monitor.predictor.predict(tds, turbidity, ph)
            
            if "error" not in result:
                print(format_prediction_output(result))
            else:
                print(f"Error: {result['error']}")
        
        else:
            print("Invalid choice. Please select 1-4.")
    
    except KeyboardInterrupt:
        print("\\n👋 Monitoring stopped by user.")
    except Exception as e:
        print(f"\\n❌ Error: {e}")

if __name__ == "__main__":
    main()
