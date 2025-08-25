"""
Streamlit web application for water quality prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.predict import WaterQualityPredictor
from src.utils.analysis_utils import validate_sensor_reading, calculate_water_quality_index, get_water_quality_guidelines

# Page configuration
st.set_page_config(
    page_title="Water Quality Prediction",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1e88e5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .quality-excellent { color: #4caf50; font-weight: bold; }
    .quality-good { color: #2196f3; font-weight: bold; }
    .quality-acceptable { color: #ff9800; font-weight: bold; }
    .quality-poor { color: #f44336; font-weight: bold; }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1e88e5;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_predictor():
    """Load the water quality predictor (cached)"""
    try:
        predictor = WaterQualityPredictor()
        if predictor.model is not None:
            return predictor
        else:
            return None
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">💧 Water Quality Prediction System</h1>', unsafe_allow_html=True)
    
    # Load predictor
    predictor = load_predictor()
    
    if predictor is None:
        st.error("❌ Model not loaded. Please ensure the model is trained first.")
        st.info("Run: `python src/models/train_model.py` to train the model.")
        return
    
    st.success("✅ Machine Learning model loaded successfully!")
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Navigation")
        page = st.selectbox(
            "Choose a page:",
            ["Single Prediction", "Batch Prediction", "Data Exploration", "Guidelines"]
        )
        
        st.markdown("---")
        st.header("📊 About")
        st.markdown("""
        This application uses a TensorFlow neural network to predict water quality 
        based on three key sensors:
        - **TDS**: Total Dissolved Solids
        - **Turbidity**: Water clarity
        - **pH**: Acidity/alkalinity
        """)
    
    # Main content based on selected page
    if page == "Single Prediction":
        single_prediction_page(predictor)
    elif page == "Batch Prediction":
        batch_prediction_page(predictor)
    elif page == "Data Exploration":
        data_exploration_page()
    elif page == "Guidelines":
        guidelines_page()

def single_prediction_page(predictor):
    """Single water sample prediction page"""
    st.header("🔬 Single Water Sample Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📈 Enter Sensor Readings")
        
        # Input fields
        tds = st.number_input(
            "TDS (Total Dissolved Solids) [mg/L]",
            min_value=0.0,
            max_value=5000.0,
            value=300.0,
            step=10.0,
            help="Measures dissolved inorganic and organic substances"
        )
        
        turbidity = st.number_input(
            "Turbidity [NTU]",
            min_value=0.0,
            max_value=100.0,
            value=2.0,
            step=0.1,
            help="Measures water clarity/cloudiness"
        )
        
        ph = st.number_input(
            "pH Level",
            min_value=4.0,
            max_value=12.0,
            value=7.0,
            step=0.1,
            help="Measures acidity/alkalinity (7 is neutral)"
        )
        
        # Predict button
        if st.button("🔍 Analyze Water Quality", type="primary"):
            analyze_single_sample(predictor, tds, turbidity, ph)
    
    with col2:
        st.subheader("📋 Quick Reference")
        
        # Display guidelines in cards
        st.markdown("""
        <div class="metric-card">
        <h4>💧 Optimal Ranges</h4>
        <ul>
        <li><strong>TDS:</strong> 50-300 mg/L</li>
        <li><strong>Turbidity:</strong> <1 NTU</li>
        <li><strong>pH:</strong> 7.0-7.5</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Sample presets
        st.subheader("🧪 Try Sample Data")
        
        samples = {
            "🏔️ Mountain Spring": (200, 0.5, 7.3),
            "🏠 Filtered Tap": (350, 1.8, 7.1),
            "🚰 City Water": (650, 4.2, 6.9),
            "🏭 Contaminated": (1800, 18.0, 5.2)
        }
        
        for name, (s_tds, s_turb, s_ph) in samples.items():
            if st.button(name):
                st.session_state.tds = s_tds
                st.session_state.turbidity = s_turb
                st.session_state.ph = s_ph
                st.experimental_rerun()

def analyze_single_sample(predictor, tds, turbidity, ph):
    """Analyze a single water sample"""
    
    # Validation
    validation = validate_sensor_reading(tds=tds, turbidity=turbidity, ph=ph)
    
    if not validation['valid']:
        st.error(f"❌ Invalid input: {'; '.join(validation['errors'])}")
        return
    
    if validation['warnings']:
        st.warning(f"⚠️ Warnings: {'; '.join(validation['warnings'])}")
    
    # Make prediction
    result = predictor.predict(tds, turbidity, ph)
    
    if "error" in result:
        st.error(f"❌ Prediction failed: {result['error']}")
        return
    
    # Display results
    st.markdown("---")
    st.subheader("🎯 Prediction Results")
    
    # Main metrics
    col1, col2, col3 = st.columns(3)
    
    quality_colors = {
        'Excellent': '#4caf50',
        'Good': '#2196f3', 
        'Acceptable': '#ff9800',
        'Poor': '#f44336'
    }
    
    quality = result['prediction']['quality_label']
    confidence = result['prediction']['confidence']
    
    with col1:
        st.metric(
            "Water Quality",
            quality,
            delta=None
        )
    
    with col2:
        st.metric(
            "Confidence",
            f"{confidence:.1%}",
            delta=None
        )
    
    with col3:
        wqi = calculate_water_quality_index(tds, turbidity, ph)
        st.metric(
            "Quality Index",
            f"{wqi:.1f}/100",
            delta=None
        )
    
    # Recommendation
    st.markdown("### 💡 Recommendation")
    st.info(result['recommendation'])
    
    # Detailed probabilities
    st.markdown("### 📊 Detailed Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Probability chart
        prob_data = result['probabilities']
        labels = list(prob_data.keys())
        values = list(prob_data.values())
        colors = [quality_colors[label] for label in labels]
        
        fig = go.Figure(data=[
            go.Bar(x=labels, y=values, marker_color=colors, text=[f'{v:.1%}' for v in values], textposition='auto')
        ])
        fig.update_layout(
            title="Quality Probabilities",
            xaxis_title="Quality Class",
            yaxis_title="Probability",
            yaxis=dict(tickformat='.0%'),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Parameter analysis
        st.markdown("**Parameter Assessment:**")
        
        # pH assessment
        if 7.0 <= ph <= 7.5:
            ph_status = "✅ Optimal"
        elif 6.5 <= ph <= 8.5:
            ph_status = "✅ Good"
        else:
            ph_status = "⚠️ Needs attention"
        
        # TDS assessment
        if tds <= 300:
            tds_status = "✅ Excellent"
        elif tds <= 600:
            tds_status = "✅ Good"
        elif tds <= 900:
            tds_status = "⚠️ Acceptable"
        else:
            tds_status = "❌ High"
        
        # Turbidity assessment
        if turbidity <= 1:
            turb_status = "✅ Excellent"
        elif turbidity <= 4:
            turb_status = "✅ Good"
        elif turbidity <= 10:
            turb_status = "⚠️ Acceptable"
        else:
            turb_status = "❌ High"
        
        st.markdown(f"""
        - **pH ({ph}):** {ph_status}
        - **TDS ({tds} mg/L):** {tds_status}  
        - **Turbidity ({turbidity} NTU):** {turb_status}
        """)

def batch_prediction_page(predictor):
    """Batch prediction page"""
    st.header("📊 Batch Water Sample Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📁 Upload Data")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload CSV file with columns: tds, turbidity, ph",
            type=['csv'],
            help="CSV file should have columns: tds, turbidity, ph"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                # Validate columns
                required_cols = ['tds', 'turbidity', 'ph']
                if not all(col in df.columns for col in required_cols):
                    st.error(f"❌ CSV must contain columns: {required_cols}")
                    return
                
                st.success(f"✅ Loaded {len(df)} samples")
                st.dataframe(df.head())
                
                if st.button("🔍 Analyze Batch", type="primary"):
                    analyze_batch_samples(predictor, df)
                    
            except Exception as e:
                st.error(f"❌ Error reading file: {e}")
    
    with col2:
        st.subheader("🧪 Generate Sample Data")
        
        n_samples = st.number_input(
            "Number of samples to generate:",
            min_value=10,
            max_value=1000,
            value=50
        )
        
        if st.button("🎲 Generate Random Samples"):
            # Generate random samples
            np.random.seed(42)
            
            sample_data = []
            for _ in range(n_samples):
                quality_target = np.random.choice([0, 1, 2, 3], p=[0.2, 0.3, 0.3, 0.2])
                
                if quality_target == 3:  # Excellent
                    tds = np.random.normal(250, 50)
                    turbidity = np.random.exponential(0.5)
                    ph = np.random.normal(7.2, 0.2)
                elif quality_target == 2:  # Good
                    tds = np.random.normal(400, 100)
                    turbidity = np.random.exponential(2.0)
                    ph = np.random.normal(7.0, 0.3)
                elif quality_target == 1:  # Acceptable
                    tds = np.random.normal(700, 150)
                    turbidity = np.random.exponential(5.0)
                    ph = np.random.normal(6.8, 0.5)
                else:  # Poor
                    tds = np.random.normal(1200, 300)
                    turbidity = np.random.exponential(12.0)
                    ph = np.random.choice([
                        np.random.normal(5.5, 0.3),
                        np.random.normal(9.0, 0.3)
                    ])
                
                sample_data.append({
                    'tds': max(50, min(3000, round(tds, 1))),
                    'turbidity': max(0.1, min(50, round(turbidity, 1))),
                    'ph': max(4.0, min(12.0, round(ph, 1)))
                })
            
            df = pd.DataFrame(sample_data)
            st.success(f"✅ Generated {len(df)} samples")
            st.dataframe(df.head(10))
            
            if st.button("🔍 Analyze Generated Samples", type="primary"):
                analyze_batch_samples(predictor, df)

def analyze_batch_samples(predictor, df):
    """Analyze batch of water samples"""
    st.markdown("---")
    st.subheader("📈 Batch Analysis Results")
    
    # Progress bar
    progress_bar = st.progress(0)
    
    results = []
    for i, row in df.iterrows():
        result = predictor.predict(row['tds'], row['turbidity'], row['ph'])
        
        if "error" not in result:
            results.append({
                'Sample': i + 1,
                'TDS': row['tds'],
                'Turbidity': row['turbidity'],
                'pH': row['ph'],
                'Quality': result['prediction']['quality_label'],
                'Confidence': result['prediction']['confidence'],
                'WQI': calculate_water_quality_index(row['tds'], row['turbidity'], row['ph'])
            })
        
        progress_bar.progress((i + 1) / len(df))
    
    results_df = pd.DataFrame(results)
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        excellent_count = len(results_df[results_df['Quality'] == 'Excellent'])
        st.metric("Excellent", excellent_count, f"{excellent_count/len(results_df)*100:.1f}%")
    
    with col2:
        good_count = len(results_df[results_df['Quality'] == 'Good'])
        st.metric("Good", good_count, f"{good_count/len(results_df)*100:.1f}%")
    
    with col3:
        acceptable_count = len(results_df[results_df['Quality'] == 'Acceptable'])
        st.metric("Acceptable", acceptable_count, f"{acceptable_count/len(results_df)*100:.1f}%")
    
    with col4:
        poor_count = len(results_df[results_df['Quality'] == 'Poor'])
        st.metric("Poor", poor_count, f"{poor_count/len(results_df)*100:.1f}%")
    
    # Visualization
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Quality distribution
        quality_counts = results_df['Quality'].value_counts()
        fig = px.pie(
            values=quality_counts.values,
            names=quality_counts.index,
            title="Quality Distribution",
            color_discrete_map={
                'Excellent': '#4caf50',
                'Good': '#2196f3',
                'Acceptable': '#ff9800', 
                'Poor': '#f44336'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Confidence distribution
        fig = px.histogram(
            results_df,
            x='Confidence',
            nbins=20,
            title="Prediction Confidence Distribution",
            color_discrete_sequence=['#1e88e5']
        )
        fig.update_layout(xaxis_title="Confidence", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
    
    # Results table
    st.subheader("📋 Detailed Results")
    
    # Add download button
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name="water_quality_results.csv",
        mime="text/csv"
    )
    
    st.dataframe(results_df, use_container_width=True)

def data_exploration_page():
    """Data exploration page"""
    st.header("📊 Data Exploration")
    
    st.markdown("""
    This page would typically show:
    - Training data distributions
    - Feature correlations
    - Model performance metrics
    - Feature importance analysis
    """)
    
    # Placeholder visualizations
    st.info("🚧 This section would contain detailed data analysis and model insights.")

def guidelines_page():
    """Water quality guidelines page"""
    st.header("📋 Water Quality Guidelines")
    
    guidelines = get_water_quality_guidelines()
    
    st.markdown(guidelines)
    
    # Interactive parameter explorer
    st.subheader("🔧 Parameter Explorer")
    
    parameter = st.selectbox("Select parameter to explore:", ["pH", "TDS", "Turbidity"])
    
    if parameter == "pH":
        st.markdown("""
        ### pH (Potential of Hydrogen)
        
        **What it measures:** The acidity or alkalinity of water on a scale of 0-14
        
        **Health impact:**
        - Too acidic (< 6.5): Can cause corrosion, metallic taste
        - Too alkaline (> 8.5): Can cause scaling, bitter taste
        
        **Standards:**
        - WHO: 6.5 - 8.5
        - EPA: 6.5 - 8.5
        - Optimal: 7.0 - 7.5
        """)
        
    elif parameter == "TDS":
        st.markdown("""
        ### TDS (Total Dissolved Solids)
        
        **What it measures:** Concentration of dissolved substances (mg/L)
        
        **Health impact:**
        - Very low: May lack essential minerals
        - Very high: May cause gastrointestinal issues, affect taste
        
        **Standards:**
        - Excellent: < 300 mg/L
        - Good: 300 - 600 mg/L  
        - Acceptable: 600 - 900 mg/L
        - Poor: > 900 mg/L
        """)
        
    else:  # Turbidity
        st.markdown("""
        ### Turbidity
        
        **What it measures:** Water clarity/cloudiness (NTU)
        
        **Health impact:**
        - High turbidity may harbor harmful microorganisms
        - Indicates filtration effectiveness
        
        **Standards:**
        - Excellent: < 1 NTU
        - Good: 1 - 4 NTU
        - Acceptable: 4 - 10 NTU
        - Poor: > 10 NTU
        """)

if __name__ == "__main__":
    main()
