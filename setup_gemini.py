#!/usr/bin/env python3
"""
Setup script for Gemini API key configuration
"""

import os
import sys

def setup_gemini_api():
    """Interactive setup for Gemini API key"""
    print("=" * 50)
    print("  Gemini API Key Setup for Water Quality System")
    print("=" * 50)
    print()
    print("To enable AI-powered water quality summaries, you need a Gemini API key.")
    print("1. Visit: https://aistudio.google.com/app/apikey")
    print("2. Create a new API key")
    print("3. Copy the API key")
    print()
    
    api_key = input("Enter your Gemini API key (or press Enter to skip): ").strip()
    
    if not api_key:
        print("Skipped Gemini setup. You can configure it later in config/config.py")
        return
    
    # Read current config
    config_path = "config/config.py"
    try:
        with open(config_path, 'r') as f:
            content = f.read()
        
        # Replace the empty API key
        updated_content = content.replace(
            "'api_key': '',  # Fill in your Gemini API key here",
            f"'api_key': '{api_key}',  # Gemini API key"
        )
        
        # Write updated config
        with open(config_path, 'w') as f:
            f.write(updated_content)
        
        print("✅ Gemini API key configured successfully!")
        print("AI-powered summaries are now enabled in the FastAPI server.")
        print()
        print("Test the API with:")
        print("uvicorn src.api.fastapi_server:app --host 0.0.0.0 --port 8000")
        
    except Exception as e:
        print(f"❌ Failed to update config: {e}")
        print(f"Please manually add your API key to {config_path}")

def test_gemini_connection():
    """Test Gemini API connection"""
    try:
        import google.generativeai as genai
        from config.config import GEMINI_CONFIG
        
        if not GEMINI_CONFIG['api_key']:
            print("❌ No API key configured")
            return False
        
        genai.configure(api_key=GEMINI_CONFIG['api_key'])
        model = genai.GenerativeModel(GEMINI_CONFIG['model_name'])
        
        # Simple test
        response = model.generate_content("Say 'Gemini API is working!' in a single sentence.")
        print(f"✅ Gemini API test successful: {response.text}")
        return True
        
    except ImportError:
        print("❌ google-generativeai package not installed")
        print("Run: pip install google-generativeai")
        return False
    except Exception as e:
        print(f"❌ Gemini API test failed: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_gemini_connection()
    else:
        setup_gemini_api()
