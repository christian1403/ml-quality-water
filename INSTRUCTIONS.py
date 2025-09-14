#!/usr/bin/env python3
"""
Complete setup and testing guide for Water Quality Prediction API with Gemini AI
"""

print("""
🧪 WATER QUALITY PREDICTION API WITH GEMINI AI INTEGRATION
===========================================================

✅ WHAT'S BEEN IMPLEMENTED:

1. Fixed ML prediction bugs (preprocessing issues)
2. Added Gemini AI integration for human-readable summaries
3. Enhanced FastAPI server with AI-powered responses
4. Created setup tools and testing utilities

📋 SETUP INSTRUCTIONS:

Step 1: Get Your Gemini API Key
------------------------------
1. Visit: https://aistudio.google.com/app/apikey
2. Sign in with your Google account  
3. Click "Create API Key"
4. Copy the generated API key

Step 2: Configure the API Key (Choose one method)
------------------------------------------------
Method A - Interactive Setup:
    python setup_gemini.py

Method B - Environment Variable:
    export GEMINI_API_KEY="your_api_key_here"

Method C - Direct Configuration:
    Edit config/config.py and replace:
    'api_key': ''
    with:
    'api_key': 'your_api_key_here'

Step 3: Test the Setup
---------------------
    python setup_gemini.py test

Step 4: Start the API Server  
----------------------------
    uvicorn src.api.fastapi_server:app --host 0.0.0.0 --port 8000

Step 5: Test the API
-------------------
    python test_api.py

🧪 MANUAL API TESTING:

Test Poor Quality Water (Your Case):
    curl -X POST "http://localhost:8000/predict" \\
         -H "Content-Type: application/json" \\
         -d '{"tds": 975, "turbidity": 8.13, "ph": 4.0}'

Test Excellent Quality Water:
    curl -X POST "http://localhost:8000/predict" \\
         -H "Content-Type: application/json" \\
         -d '{"tds": 200, "turbidity": 0.5, "ph": 7.2}'

📊 API ENDPOINTS:

- GET  /              - API information
- GET  /health        - Health check
- POST /predict       - Single prediction with AI summary
- POST /predict/batch - Batch predictions
- POST /analyze       - Comprehensive analysis
- GET  /guidelines    - Water quality standards

🔧 FEATURES:

✅ Fixed Prediction: Now correctly identifies poor quality water
✅ AI Summaries: Human-readable explanations via Gemini AI
✅ Fallback Handling: Works with or without Gemini API key
✅ Comprehensive Analysis: Technical + AI-powered insights
✅ Validation: Input validation and error handling
✅ Documentation: Interactive API docs at /docs

🎯 EXAMPLE WITH GEMINI (After Setup):

Input: TDS=975, Turbidity=8.13, pH=4.0

Response:
{
  "prediction": {
    "quality_label": "Poor",
    "confidence": 92.0%
  },
  "summary": "This water sample shows concerning quality issues that make it unsafe for drinking. The extremely acidic pH level of 4.0 and high dissolved solids content indicate the need for proper water treatment before consumption. I recommend using a water filtration system or seeking an alternative water source."
}

📝 NEXT STEPS:

1. Configure your Gemini API key
2. Test the API functionality
3. Integrate with your applications
4. Deploy to production environment

💡 Need Help?
- Check GEMINI_SETUP.md for detailed instructions
- Run test_api.py for automated testing
- Visit http://localhost:8000/docs for interactive API documentation
""")
