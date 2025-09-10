# Water Quality API - Vercel Deployment Guide

This guide will help you deploy your FastAPI water quality prediction service to Vercel.

## 🚀 Quick Deploy to Vercel

### Prerequisites

1. **Vercel Account**: Sign up at [vercel.com](https://vercel.com)
2. **GitHub Repository**: Your code should be in a GitHub repository
3. **Gemini AI API Key** (optional): For AI-powered summaries

### Step 1: Environment Variables

1. In your Vercel dashboard, go to your project settings
2. Navigate to "Environment Variables"
3. Add the following variables:

```
GEMINI_API_KEY=your_gemini_api_key_here
ENVIRONMENT=production
```

### Step 2: Deploy

#### Option A: Deploy from GitHub (Recommended)

1. Connect your GitHub repository to Vercel
2. Vercel will automatically detect the FastAPI application
3. Deploy with default settings

#### Option B: Deploy with Vercel CLI

```bash
# Install Vercel CLI
npm i -g vercel

# Login to Vercel
vercel login

# Deploy from project root
vercel

# For production deployment
vercel --prod
```

### Step 3: Test Your Deployment

Once deployed, your API will be available at `https://your-project.vercel.app`

Test endpoints:
- `GET /` - API information
- `GET /health` - Health check
- `POST /predict/demo` - Demo prediction (always works)
- `POST /predict` - ML prediction (requires model files)

## 📡 API Endpoints

### Demo Prediction (Always Available)
```bash
curl -X POST "https://your-project.vercel.app/predict/demo" \
  -H "Content-Type: application/json" \
  -d '{
    "tds": 250,
    "turbidity": 1.5,
    "ph": 7.2
  }'
```

### Health Check
```bash
curl "https://your-project.vercel.app/health"
```

### Get Guidelines
```bash
curl "https://your-project.vercel.app/guidelines"
```

## ⚙️ Configuration Files

The deployment uses these key files:

- `api/index.py` - Entry point for Vercel
- `vercel.json` - Vercel configuration
- `requirements.txt` - Python dependencies
- `.env.example` - Environment variables template

## 🔧 Local Development

To run locally (same as before):

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn src.api.fastapi_server:app --host 0.0.0.0 --port 8000
```

## 📝 Notes

- **Model Files**: Large ML model files might not be included in the deployment due to size limits
- **Demo Mode**: The API includes a demo endpoint that works without ML models
- **Gemini AI**: Optional feature for AI-powered summaries
- **Cold Starts**: First request might be slower due to Vercel's cold start

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Check that all paths in `api/index.py` are correct
2. **Missing Dependencies**: Ensure `requirements.txt` includes all needed packages
3. **Model Loading Fails**: Use `/predict/demo` endpoint for testing
4. **Environment Variables**: Check they are properly set in Vercel dashboard

### Logs

Check deployment logs in Vercel dashboard:
- Go to your project
- Click on a deployment
- View "Function Logs" tab

## 📈 Monitoring

Monitor your API using:
- Vercel Analytics (built-in)
- `/health` endpoint for basic health checks
- Custom logging in your application

## 🔒 Security

- Environment variables are encrypted by Vercel
- HTTPS is enabled by default
- Consider rate limiting for production use

---

**Need help?** Check the [Vercel Documentation](https://vercel.com/docs) or create an issue in this repository.
