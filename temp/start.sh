#!/bin/bash

echo "Starting ClipIt..."

# Use Python 3.12 for better package compatibility
PYTHON="/opt/homebrew/bin/python3.12"

# Setup backend
cd backend

# Create a fresh venv with Python 3.12
rm -rf venv
$PYTHON -m venv venv
source venv/bin/activate

# Install backend dependencies step by step
echo "Installing backend dependencies..."
pip install --upgrade pip
pip install setuptools wheel
pip install flask flask-cors numpy 
pip install opencv-python
pip install mediapipe>=0.10.5
pip install moviepy==1.0.3
pip install scenedetect
pip install sentence-transformers
pip install google-generativeai
pip install scikit-learn

# Ensure protobuf compatibility at the end
pip uninstall -y protobuf
pip install protobuf==4.25.7

# Create required directories
mkdir -p uploads snippets

# Start backend
echo "Starting backend server..."
python app.py &
BACKEND_PID=$!

# Start frontend
cd ../frontend

# Install frontend dependencies
echo "Installing frontend dependencies..."
npm install

# Start frontend on port 3001 explicitly
echo "Starting frontend server..."
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

echo "ClipIt is running!"
echo "Backend: http://localhost:5001"
echo "Frontend: http://localhost:3001"
echo "Press Ctrl+C to stop"

# Wait for both processes
wait