# config_example.py — copy to config.py and fill with real values.
NEO4J_URI = "bolt://your-neo4j-uri"
NEO4J_USER = "your-username"
NEO4J_PASSWORD = "your-password"

OPENAI_API_KEY = "your-openai-api-key"

PINECONE_API_KEY = "your-pinecone-api-key"
PINECONE_ENV = "us-east-1"   # example
PINECONE_INDEX_NAME = "index-name"
PINECONE_VECTOR_DIM = 1536       # adjust to embedding model used (text-embedding-3-large ~ 3072? check your model); we assume 1536 for common OpenAI models — change if needed.
