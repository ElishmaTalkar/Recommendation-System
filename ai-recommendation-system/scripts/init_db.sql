-- Initialize pgvector extension and create indexes

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create items table is handled by SQLAlchemy
-- This script is for additional initialization if needed

-- Set default search parameters for better performance
ALTER DATABASE recommendations SET maintenance_work_mem = '512MB';
