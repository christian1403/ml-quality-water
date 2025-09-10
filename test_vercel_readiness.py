#!/usr/bin/env python3
"""
Test script for Vercel deployment readiness
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_import():
    """Test if all imports work"""
    try:
        from api.index import app
        print("✅ API import successful")
        return True
    except Exception as e:
        print(f"❌ API import failed: {e}")
        return False

def test_fastapi_app():
    """Test FastAPI app initialization"""
    try:
        from src.api.fastapi_server import app
        print("✅ FastAPI app initialization successful")
        return True
    except Exception as e:
        print(f"❌ FastAPI app initialization failed: {e}")
        return False

def test_config():
    """Test configuration loading"""
    try:
        from config.config import GEMINI_CONFIG, QUALITY_LABELS
        print("✅ Configuration loading successful")
        print(f"   - Gemini API key configured: {'Yes' if GEMINI_CONFIG['api_key'] else 'No'}")
        print(f"   - Quality labels available: {len(QUALITY_LABELS)} classes")
        return True
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Vercel deployment readiness...\n")
    
    tests = [
        ("Import Test", test_import),
        ("FastAPI App Test", test_fastapi_app),
        ("Configuration Test", test_config),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"Running {test_name}...")
        if test_func():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Ready for Vercel deployment.")
        print("\nNext steps:")
        print("1. Push your code to GitHub")
        print("2. Connect GitHub repo to Vercel")
        print("3. Set environment variables in Vercel dashboard")
        print("4. Deploy!")
    else:
        print("⚠️  Some tests failed. Please fix the issues before deploying.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
