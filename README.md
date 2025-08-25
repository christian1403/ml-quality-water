# Water Quality Prediction with Machine Learning

## Project Overview
This project uses machine learning to predict water quality for consumption based on three key sensor measurements:
- **TDS (Total Dissolved Solids)**: Measures dissolved inorganic and organic substances
- **Turbidity**: Measures water clarity/cloudiness
- **pH**: Measures acidity/alkalinity levels

## Technology Stack
- **TensorFlow**: Deep learning framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning utilities
- **Matplotlib/Seaborn**: Data visualization

## Project Structure
```
ml-quality-water/
├── data/                   # Dataset storage
├── models/                 # Trained models
├── notebooks/              # Jupyter notebooks for exploration
├── src/                    # Source code
│   ├── data_processing/    # Data preprocessing utilities
│   ├── models/             # Model architectures
│   └── utils/              # Helper functions
├── tests/                  # Unit tests
└── config/                 # Configuration files
```

## Water Quality Standards
Based on WHO and EPA guidelines:
- **pH**: Acceptable range 6.5-8.5
- **TDS**: < 500 mg/L (excellent), 500-1000 mg/L (good), > 1000 mg/L (poor)
- **Turbidity**: < 1 NTU (excellent), 1-4 NTU (acceptable), > 4 NTU (poor)

## Getting Started
1. Create a virtual environment: 
```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
```
2. Install dependencies: `pip install -r requirements.txt`
3. Run data generation: `python src/data_processing/generate_data.py`
4. Train model: `python main.py --train`
5. Make predictions: `python main.py --predict [TDS] [Turbidity] [pH]`
6. Analyze Data: `python main.py --analyze`
7. Interactive: `python main.py --interactive`
8. Run Streamlit:
```bash
    streamlit run streamlit.py
```
9. Run FastApi:
```bash
    uvicorn src.api.fastapi_server:app --host 0.0.0.0 --port 8000
```

## Model Performance Metrics
The model will be evaluated using:
- Accuracy
- Precision/Recall
- F1-Score
- Confusion Matrix
- ROC-AUC Score
