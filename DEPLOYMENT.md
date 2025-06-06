# ClipIt Deployment Guide

This guide covers deploying the separated ClipIt backend and frontend repositories.

## Repository Structure

The project has been split into two repositories:

1. **Backend** (`repos/clipit-backend/`) - Python Flask API for fly.io
2. **Frontend** (`repos/clipit-frontend/`) - Next.js React app for Vercel

## Backend Deployment (Fly.io)

### 1. Create GitHub Repository
```bash
cd repos/clipit-backend
gh repo create clipit-backend --public --source=. --remote=origin --push
```

### 2. Deploy to Fly.io
```bash
# Install fly CLI if not already installed
# curl -L https://fly.io/install.sh | sh

# Login to fly.io
flyctl auth login

# Create and deploy the app
flyctl launch --no-deploy

# Set environment variables
flyctl secrets set GEMINI_API_KEY=your_actual_gemini_api_key_here

# Deploy the app
flyctl deploy
```

### 3. Note Your Backend URL
After deployment, note your backend URL (e.g., `https://clip-it.fly.dev`)

## Frontend Deployment (Vercel)

### 1. Create GitHub Repository
```bash
cd repos/clipit-frontend
gh repo create clipit-frontend --public --source=. --remote=origin --push
```

### 2. Deploy to Vercel
1. Go to [vercel.com](https://vercel.com) and sign in
2. Click "Import Project" and select your `clipit-frontend` repository
3. Configure environment variables:
   - `NEXT_PUBLIC_API_URL` = Your fly.io backend URL (e.g., `https://clip-it.fly.dev`)
4. Deploy

## Environment Variables

### Backend (.env)
```bash
GEMINI_API_KEY=your_gemini_api_key_here
PORT=8080
```

### Frontend (.env.local)
```bash
NEXT_PUBLIC_API_URL=https://your-backend-url.fly.dev
```

## Post-Deployment

1. Test the backend health endpoint: `https://your-backend-url.fly.dev/health`
2. Test the frontend at your Vercel URL
3. Upload a test video to ensure the integration works

## Repository Links

After creating the repositories, you'll have:
- Backend: `https://github.com/yourusername/clipit-backend`
- Frontend: `https://github.com/yourusername/clipit-frontend`