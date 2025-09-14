import sys
import os

# Add the project root to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.fastapi_server import app

# Vercel will automatically detect the variable named "app"
# No need to run uvicorn manually