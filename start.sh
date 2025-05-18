#!/bin/bash

echo "Starting ClipIt..."

# Look for Python 3 in common locations
PYTHON=$(which python3 || which python3.11 || which python3.10 || which python3.9 || which python3.8 || which python)
if [ -z "$PYTHON" ]; then
  echo "Error: Python 3 is required but not found"
  exit 1
fi

# Setup backend
cd backend

# Create a virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
  echo "Creating virtual environment..."
  $PYTHON -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install backend dependencies
echo "Installing backend dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Make sure API key is set
if [ -z "$GEMINI_API_KEY" ]; then
  if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
  else
    echo -n "Please enter your Gemini API key: "
    read GEMINI_API_KEY
    echo "GEMINI_API_KEY=$GEMINI_API_KEY" > .env
    export GEMINI_API_KEY
  fi
fi

# Create required directories
mkdir -p uploads snippets

# Start backend
echo "Starting backend server..."
python app.py &
BACKEND_PID=$!

# Start frontend
cd ../frontend

# Install frontend dependencies if needed
if [ ! -d "node_modules" ]; then
  echo "Installing frontend dependencies..."
  npm install
fi

# Start frontend on port 3001 explicitly
echo "Starting frontend server..."
export NEXT_PUBLIC_API_URL="http://localhost:5001"
npx next dev -p 3001 &
FRONTEND_PID=$!

# Function to handle script termination
cleanup() {
    echo "Shutting down servers..."
    kill $BACKEND_PID
    kill $FRONTEND_PID
    exit
}

# Register the cleanup function for when script is terminated
trap cleanup SIGINT SIGTERM

echo ""
echo "ClipIt is running!"
echo "Backend: http://localhost:5001"
echo "Frontend: http://localhost:3001"
echo "Press Ctrl+C to stop"

# Wait for both processes
wait