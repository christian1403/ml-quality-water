# Gemini AI Integration Setup

This project now includes AI-powered water quality summaries using Google's Gemini API.

## Setup Instructions

### Option 1: Interactive Setup (Recommended)
```bash
python setup_gemini.py
```

### Option 2: Environment Variable
```bash
export GEMINI_API_KEY="your_api_key_here"
```

### Option 3: Direct Configuration
Edit `config/config.py` and replace the empty API key:
```python
GEMINI_CONFIG = {
    'api_key': 'your_api_key_here',  # Add your key here
    'model_name': 'gemini-1.5-flash',
    'temperature': 0.7,
    'max_output_tokens': 500
}
```

## Getting Your Gemini API Key

1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the generated key

## Testing the Integration

### Start the FastAPI Server
```bash
uvicorn src.api.fastapi_server:app --host 0.0.0.0 --port 8000
```

### Test API Endpoints
```bash
python test_api.py
```

### Manual API Test
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"tds": 975, "turbidity": 8.13, "ph": 4.0}'
```

## Features

- **AI-Powered Summaries**: Human-readable explanations of water quality results
- **Technical Analysis**: Detailed parameter breakdown
- **Safety Recommendations**: Actionable advice for consumers
- **Fallback Handling**: Works with or without Gemini API key

## Example Response

```json
{
  "input": {
    "tds": 975,
    "turbidity": 8.13,
    "ph": 4.0
  },
  "prediction": {
    "quality_class": 0,
    "quality_label": "Poor",
    "confidence": 0.92
  },
  "probabilities": {
    "Poor": 0.92,
    "Acceptable": 0.079,
    "Good": 0.0006,
    "Excellent": 0.0001
  },
  "recommendation": "Poor water quality. Treatment required before consumption.",
  "summary": "This water sample shows concerning quality issues that make it unsafe for drinking. The extremely acidic pH level of 4.0 and high dissolved solids content of 975 mg/L indicate the need for proper water treatment before consumption. I recommend using a water filtration system or seeking an alternative water source to ensure your safety.",
  "timestamp": "2025-08-31T16:30:00"
}
```
