#!/usr/bin/env python3
"""
Demo script to test the FastAPI server with Gemini AI integration
"""

import requests
import json
import time

def test_api_endpoint(base_url="http://localhost:8000"):
    """Test the water quality prediction API"""
    
    print("🧪 Testing Water Quality Prediction API with Gemini AI")
    print("=" * 60)
    
    # Test cases
    test_cases = [
        {
            "name": "Excellent Quality Water",
            "data": {"tds": 200, "turbidity": 0.5, "ph": 7.2},
            "expected": "Excellent"
        },
        {
            "name": "Poor Quality Water (Your Test Case)",
            "data": {"tds": 975, "turbidity": 8.13, "ph": 4.0},
            "expected": "Poor"
        },
        {
            "name": "Very Poor Quality Water",
            "data": {"tds": 1500, "turbidity": 15.0, "ph": 10.5},
            "expected": "Poor"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test {i}: {test_case['name']}")
        print(f"Input: TDS={test_case['data']['tds']}, Turbidity={test_case['data']['turbidity']}, pH={test_case['data']['ph']}")
        
        try:
            response = requests.post(f"{base_url}/predict", json=test_case['data'])
            
            if response.status_code == 200:
                result = response.json()
                prediction = result['prediction']
                
                print(f"✅ Prediction: {prediction['quality_label']} ({prediction['confidence']:.1%} confidence)")
                
                if 'summary' in result:
                    print(f"🤖 AI Summary: {result['summary']}")
                else:
                    print("❌ No AI summary (Gemini not configured)")
                
                # Verify expected result
                if prediction['quality_label'] == test_case['expected']:
                    print("✅ Result matches expectation")
                else:
                    print(f"⚠️  Expected {test_case['expected']}, got {prediction['quality_label']}")
                    
            else:
                print(f"❌ API Error: {response.status_code} - {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("❌ Connection Error: Make sure the FastAPI server is running")
            print("Start server with: uvicorn src.api.fastapi_server:app --host 0.0.0.0 --port 8000")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("-" * 60)

def check_server_status(base_url="http://localhost:8000"):
    """Check if the API server is running"""
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            health = response.json()
            print(f"🟢 Server Status: {health['status']}")
            print(f"🤖 Model Status: {health['model_status']}")
            return True
        else:
            print(f"🔴 Server Error: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("🔴 Server not running")
        return False

if __name__ == "__main__":
    print("🚀 Water Quality API Testing Tool")
    print()
    
    # Check server status first
    if check_server_status():
        print("\n" + "=" * 60)
        test_api_endpoint()
    else:
        print("\n📝 To start the server:")
        print("1. Configure Gemini API key: python setup_gemini.py")
        print("2. Start server: uvicorn src.api.fastapi_server:app --host 0.0.0.0 --port 8000")
        print("3. Run this test again: python test_api.py")
