# AI-Powered Semantic Search & Recommendation Engine

![React](https://img.shields.io/badge/React-18-61DAFB.svg)
![TypeScript](https://img.shields.io/badge/TypeScript-5.3-3178C6.svg)
![Vite](https://img.shields.io/badge/Vite-5.1-646CFF.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-009688.svg)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-17-336791.svg)
![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)

A full-stack application featuring a modern React frontend and a powerful AI-driven backend for semantic search and product recommendations.

## 🌟 Project Overview

This repository contains two main components:

1.  **Frontend (`/`)**: A responsive web interface built with **React**, **TypeScript**, and **Tailwind CSS**.
2.  **Backend (`/ai-recommendation-system`)**: An advanced AI recommendation engine using **FastAPI**, **pgvector**, and **Sentence Transformers**.

---

## 🚀 Quick Start

### 1. Backend Setup (AI Engine)
The backend handles data ingestion, vector embeddings, and recommendation logic.

```bash
cd ai-recommendation-system

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start API
uvicorn src.api.routes:app --reload
```
> **Detailed Backend Documentation**: [Read the full guide here](./ai-recommendation-system/README.md)

### 2. Frontend Setup (Web UI)
The frontend provides the user interface for searching and viewing recommendations.

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```
The app will be available at `http://localhost:5173`.

---

## 🏗️ Architecture

### Frontend
- **Framework**: React + Vite
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **State Management**: React Hooks

### Backend
- **API**: FastAPI (Python)
- **Database**: PostgreSQL with `pgvector` extension
- **AI Models**: `sentence-transformers/all-MiniLM-L6-v2`
- **RAG Pipeline**: Claude 3 (Anthropic) for re-ranking

## 📂 Repository Structure

```
.
├── ai-recommendation-system/   # Python Backend & AI Logic
│   ├── src/                    # Source code
│   ├── scripts/                # Data ingestion scripts
│   └── docker-compose.yml      # DB deployment
├── src/                        # Frontend React Source
├── public/                     # Static assets
├── package.json                # Frontend dependencies
├── tsconfig.json               # TypeScript config
└── README.md                   # Project documentation
```

## ✨ Key Features

- **Semantic Search**: Find products by meaning, not just keywords (e.g., "warm winter clothes" finds down jackets).
- **Hybrid Recommendations**: Combines content similarity with collaborative filtering.
- **RAG Explanations**: Uses LLMs (`Claude`) to explain *why* a product was recommended.
- **Modern UI**: Clean, responsive interface with instant search results.