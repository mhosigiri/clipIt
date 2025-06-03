# ClipIt

ClipIt is an AI-powered video clip extraction tool that uses Google's Gemini AI to find and extract the most interesting or relevant clips from your videos based on your prompts.

## Features

- Upload videos and extract the most relevant clips based on your text prompts
- Powered by Google's Gemini AI for intelligent clip selection
- Scene detection and content analysis using computer vision
- User feedback collection for continuous improvement
- Usage statistics tracking

## Project Structure

- `frontend/`: Next.js frontend application
- `backend/`: Python Flask backend application

## Prerequisites

- Docker and Docker Compose
- Google Gemini API key (Get one from https://makersuite.google.com/app/apikey)

## Local Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/clipit.git
   cd clipit
   ```

2. Create a `.env` file in the project root (copy from `.env.sample`):
   ```bash
   cp .env.sample .env
   ```

3. Edit the `.env` file and add your Gemini API key:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

4. Start the application using Docker Compose:
   ```bash
   docker-compose up
   ```

5. The application will be available at:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8080

## Debugging

For development with hot-reloading and debugging capabilities:

```bash
docker-compose -f docker-compose.debug.yml up
```

## Deployment Options

### Deploying with Docker Compose

For a simple production deployment:

1. Build and start the containers:
   ```bash
   docker-compose up -d --build
   ```

2. Access the application at:
   - Frontend: http://your-server-ip:3000
   - Backend API: http://your-server-ip:8080

### Deploying Frontend to Vercel

1. Push your code to a GitHub repository

2. Go to https://vercel.com and create a new account or sign in

3. Import your GitHub repository

4. Configure the project:
   - Framework Preset: Next.js
   - Root Directory: frontend
   - Environment Variables: Add `NEXT_PUBLIC_API_URL` pointing to your backend URL

5. Deploy the project

### Deploying Backend to Fly.io

1. Install the Fly.io CLI:
   ```bash
   curl -L https://fly.io/install.sh | sh
   ```

2. Log in to Fly.io:
   ```bash
   fly auth login
   ```

3. Navigate to the backend directory:
   ```bash
   cd backend
   ```

4. Create a new Fly.io app:
   ```bash
   fly launch
   ```
   - This will detect the Python app and create a fly.toml file

5. Set the Gemini API key:
   ```bash
   fly secrets set GEMINI_API_KEY=your_api_key_here
   ```

6. Deploy the backend:
   ```bash
   fly deploy
   ```

7. Once deployed, update the frontend's `NEXT_PUBLIC_API_URL` to point to your Fly.io app URL.

## License

MIT