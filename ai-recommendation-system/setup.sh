#!/bin/bash

# Quick setup script for AI Recommendation System

echo "🚀 AI-Powered Recommendation System - Quick Setup"
echo "=================================================="

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker Desktop first."
    exit 1
fi

echo "✓ Docker found"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.11+."
    exit 1
fi

echo "✓ Python 3 found"

# Create .env if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cp .env.example .env
    echo "⚠️  Please edit .env and add your ANTHROPIC_API_KEY"
fi

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Download NLP models
echo "📥 Downloading NLP models..."
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"

# Generate sample data
echo "📊 Generating sample data..."
python scripts/generate_sample_data.py

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env and add your ANTHROPIC_API_KEY (optional)"
echo "2. Start services: docker-compose up -d"
echo "3. Ingest data: python scripts/ingest_data.py --input data/raw/sample_products.csv --init-db --create-index"
echo "4. Run API: uvicorn src.api.routes:app --reload"
echo "5. Visit http://localhost:8000/docs for API documentation"
echo ""
echo "Or run everything with Docker:"
echo "  docker-compose up -d"
echo ""
