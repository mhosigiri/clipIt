#!/bin/bash

echo "Starting ClipIt..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
  echo "Error: Docker is required but not found. Please install Docker first."
  exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
  echo "Error: Docker Compose is required but not found. Please install Docker Compose first."
  exit 1
fi

# Check for .env file, create if it doesn't exist
if [ ! -f ".env" ]; then
  if [ -f ".env.sample" ]; then
    echo "Creating .env file from sample..."
    cp .env.sample .env
    echo "Please edit the .env file and add your Gemini API key."
    exit 1
  else
    echo "Error: .env.sample file not found. Please create a .env file with your Gemini API key."
    echo "GEMINI_API_KEY=your_api_key_here" > .env
    echo "Please edit the .env file and add your Gemini API key."
    exit 1
  fi
fi

# Check if GEMINI_API_KEY is set in .env
if ! grep -q "GEMINI_API_KEY=" .env || grep -q "GEMINI_API_KEY=your_gemini_api_key_here" .env; then
  echo "Error: GEMINI_API_KEY is not set in .env file."
  echo "Please edit the .env file and add your Gemini API key."
  exit 1
fi

# Choose development or production mode
echo "Select mode:"
echo "1) Development (with hot-reloading)"
echo "2) Production"
read -p "Enter choice [1-2]: " mode

case $mode in
  1)
    echo "Starting ClipIt in development mode..."
    docker-compose -f docker-compose.debug.yml up --build
    ;;
  2)
    echo "Starting ClipIt in production mode..."
    docker-compose up --build
    ;;
  *)
    echo "Invalid choice. Exiting."
    exit 1
    ;;
esac