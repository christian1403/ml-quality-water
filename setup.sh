#!/bin/bash

# Water Quality ML Project Setup Script
echo "🌊 Setting up Water Quality Prediction System..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

echo "✅ Python 3 found"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📥 Installing Python packages..."
pip install -r requirements.txt

echo "✅ Dependencies installed successfully!"

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p data
mkdir -p models
mkdir -p logs

echo "✅ Project structure created"

# Generate initial dataset
echo "🔬 Generating initial water quality dataset..."
python src/data_processing/generate_data.py

if [ $? -eq 0 ]; then
    echo "✅ Dataset generated successfully!"
else
    echo "❌ Failed to generate dataset"
    exit 1
fi

# Train initial model
echo "🤖 Training initial ML model..."
python src/models/train_model.py

if [ $? -eq 0 ]; then
    echo "✅ Model trained successfully!"
else
    echo "❌ Failed to train model"
    exit 1
fi

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📋 Available commands:"
echo "  python main.py --help                    Show all available options"
echo "  python main.py --predict 300 1.5 7.2    Make a single prediction"
echo "  python main.py --interactive             Interactive prediction mode"
echo "  python main.py --analyze                 Analyze dataset"
echo ""
echo "🌐 Web applications:"
echo "  streamlit run src/app/streamlit_app.py   Launch web interface"
echo "  python src/api/fastapi_server.py         Launch API server"
echo ""
echo "📓 Jupyter notebook:"
echo "  jupyter notebook notebooks/water_quality_prediction.ipynb"
echo ""
echo "🧪 Run tests:"
echo "  python tests/test_water_quality.py"
echo ""
echo "Happy water quality prediction! 💧"
