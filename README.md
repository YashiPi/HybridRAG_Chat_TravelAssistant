# Hybrid AI Travel Assistant Challenge

## Goal

Build and debug a hybrid AI assistant that answers travel queries using:

- **Pinecone** (semantic vector DB)
- **Neo4j** (graph context)
- **OpenAI Chat Models**

This project was extended to include an advanced, production-grade architecture featuring a multi-step RAG (Retrieval-Augmented Generation) pipeline, intelligent caching, and context pre-processing.

## Features & Improvements

This implementation goes beyond the base requirements and includes:

1.  **Semantic Caching:** A separate Pinecone index (`vietnam-travel-cache`) is used to store the answers to common questions. This drastically reduces latency and API costs.
2.  **LLM Query Normalization (Groq):** User queries are first normalized by a high-speed Groq LLM (Llama3-8b) into a standardized tag (e.g., `vietnam | 4 day | romantic`). This "intent tag" is used as the key for the semantic cache, allowing it to catch differently-phrased questions with the same meaning.
3.  **LLM Context Summarization (Groq):** On a cache miss, the raw, noisy context from Pinecone and Neo4j is _not_ sent directly to the final model. Instead, it is "pre-processed" by a Groq LLM to create a clean, concise summary paragraph.
4.  **Chain-of-Thought (CoT) Prompting:** The final prompt to OpenAI's `gpt-4o-mini` uses CoT reasoning, instructing the model to "think step-by-step" to analyze the clean summary and build a more logical and accurate response.
5.  **Token Usage Monitoring:** The script will print the number of prompt tokens sent to the final OpenAI model, clearly demonstrating the cost-saving benefits of the summarization step.

## Setup & Running

### 1. Set Your API Keys in `config.py`

Copy `config.py.sample` to `config.py` and fill in all the required API keys:

- `OPENAI_API_KEY`
- `PINECONE_API_KEY`
- `GROQ_API_KEY` (from [GroqCloud](https://console.groq.com/keys))
- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` (from your AuraDB or Desktop instance)

### 2. Create Your Pinecone Indexes

You must create **two** separate indexes in your Pinecone dashboard with the same vector dimension (e.g., 1536 for `text-embedding-3-small`):

1.  **Main RAG Index:** `vietnam-travel`
2.  **Cache Index:** `vietnam-travel-cache`

### 3. Create a Virtual Environment & Install Dependencies

It's highly recommended to use the provided `requirements.txt` file.

```bash
# Create and activate the environment
python -m venv venv
source venv/bin/activate  # On macOS/Linux
.\venv\Scripts\activate    # On Windows

# Install all dependencies
pip install -r requirements.txt
```

### 4. Load the Data

```bash
# 1. Load data into your Neo4j database
   python load_to_neo4j.py

# 2. (Optional) Run the visualizer to confirm graph
   python visualize_graph.py

# 3. Create and upload vector embeddings to your main RAG index
   python pinecone_upload.py
```

### 5. Run the Hybrid Chat Assistant

Now you are ready to run the main application

```bash
python hybrid_chat.py
```

Ask it a question. The first time you ask, it will be a "Cache Miss" and run the full pipeline. The second time you (or anyone) asks a semantically similar question, it will be a "Cache Hit" and return the answer instantly.

**Example Test Query:** create a romantic 4 day itinerary for Vietnam

## Deliverables

- Working Scripts
  - pinecone_upload.py
  - hybrid_chat.py (fully upgraded version)
- Write-up:
  - improvements.md (detailing the fixes and advanced features)
- Video:
  - https://www.loom.com/share/636e24395115438e9ea66964e4647fa8
- Document:
  - https://docs.google.com/document/d/10Q0Bl5NnJZkrUWa8aKfktvEl9DHKQKsKuSkvs5mh3b4/edit?usp=sharing
