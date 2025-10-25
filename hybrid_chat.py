# hybrid_chat.py
import json
from typing import List
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from neo4j import GraphDatabase
import config
from groq import Groq
import re



# -----------------------------
# Config
# -----------------------------
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
LLM_NORMALIZER_MODEL = "llama-3.1-8b-instant"
LLM_SUMMARIZER_MODEL = "llama-3.1-8b-instant"
TOP_K = 5

INDEX_NAME = config.PINECONE_INDEX_NAME
CACHE_INDEX_NAME = "vietnam-travel-cache"

# -----------------------------
# Initialize clients
# -----------------------------
client = OpenAI(api_key = config.OPENAI_API_KEY)
pc = Pinecone(api_key = config.PINECONE_API_KEY)
groq_client = Groq(api_key = config.GROQ_API_KEY)

# Connect to Pinecone index
if CACHE_INDEX_NAME not in pc.list_indexes().names():
    print(f"Creating cache index: {CACHE_INDEX_NAME}")
    pc.create_index(
        name=CACHE_INDEX_NAME,
        dimension=config.PINECONE_VECTOR_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east1-gcp")
    )

print(f"Connecting to Pinecone index: {INDEX_NAME}")
index = pc.Index(INDEX_NAME)
cache_index = pc.Index(CACHE_INDEX_NAME)

# Set a similarity threshold
CACHE_THRESHOLD = 0.95

# Connect to Neo4j
driver = GraphDatabase.driver(
    config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
)

# -----------------------------
# Caching & Normalization
# -----------------------------

# a simple in-memory cache
embedding_cache = {}

LLM_NORMALIZE_PROMPT = """You are an ultra-fast query tagging engine. Your sole purpose is to analyze a user's travel query and extract three specific entities: LOCATION, DURATION, and TOPIC.

You MUST follow these rules:
1.  **LOCATION:** The primary city or country. If not specified, use 'na'.
2.  **DURATION:** The time (e.g., '2 day', '1 week'). If not specified, use 'na'.
3.  **TOPIC:** The main intent (e.g., 'itinerary', 'sightings', 'food', 'hotel', 'romantic', 'adventure', 'beach'). If it's a general request, use 'general'.
4.  **FORMAT:** You MUST reply with ONLY a single line in the format: `LOCATION | DURATION | TOPIC`.
5.  **DO NOT** add any explanation, preamble, or apologies.
"""

LLM_SUMMARY_PROMPT = """You are a highly efficient summarization engine. Your job is to read a user's query and a set of raw, messy context data (from a vector database and a graph database).
Your sole purpose is to synthesize this raw data into a single, clean, concise paragraph.
This summary will be used by another AI to generate the final answer, so it MUST be:
1.  **Relevant:** Only include information that directly helps answer the user's query.
2.  **Concise:** Do not use filler words. Get straight to the facts.
3.  **Factual:** Do not invent information. Stick to the provided context.
4.  **Cite IDs:** Preserve and include any node IDs (e.g., 'hotel_301', 'attraction_205') in parentheses after the item name.
"""

# Fallback normalizer in case Groq fails
STOP_WORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'can', 'create', 'find', 'for', 
    'from', 'give', 'has', 'have', 'he', 'her', 'him', 'his', 'how', 'i', 'in', 
    'is', 'it', 'its', 'make', 'me', 'of', 'on', 'or', 'or', 'please', 'provide', 
    'see', 'she', 'that', 'the', 'their', 'them', 'then', 'there', 'these', 'they', 
    'this', 'to', 'was', 'what', 'when', 'where', 'which', 'who', 'will', 'with', 
    'would', 'you', 'your', 'tell', 'about'
}

def normalize_query_fallback(query: str) -> str:
    """Cleans a query to focus on its core semantic topic by removing punctuation and stop words."""
    query = query.lower()
    query = re.sub(r'[^\w\s]', '', query) # Remove punctuation
    words = query.split()
    filtered_words = [word for word in words if word not in STOP_WORDS]
    # # Sort the words to create a consistent, order-independent key
    # filtered_words.sort()
    return " ".join(filtered_words)

# LLM Normalizer function
def normalize_query_llm(query: str) -> str:
    """Normalizes a query using a fast LLM (Groq) for semantic cache matching."""
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": LLM_NORMALIZE_PROMPT},
                {"role": "user", "content": f"Query: \"{query}\""}
            ],
            model=LLM_NORMALIZER_MODEL,
            temperature=0.0,
            max_tokens=50
        )
        normalized_query = chat_completion.choices[0].message.content.strip()
        
        # Validation check: Did the LLM follow instructions?
        if '|' not in normalized_query or len(normalized_query.split('|')) != 3:
            print("DEBUG: LLM normalization failed format, using fallback.")
            return normalize_query_fallback(query)
            
        return normalized_query
    except Exception as e:
        print(f"DEBUG: Groq API call failed ({e}), using fallback.")
        return normalize_query_fallback(query)


# -----------------------------
# Helper functions
# -----------------------------

def embed_text(text: str) -> List[float]:
    if text in embedding_cache:
        print("DEBUG: Using cached embedding.")
        return embedding_cache[text]

    """Get embedding for a text string."""
    print("DEBUG: Calling OpenAI for new embedding.")
    resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
    embedding = resp.data[0].embedding

    embedding_cache[text] = embedding
    return embedding


def pinecone_query(query_text: str, top_k=TOP_K):
    """Query Pinecone RAG index using embedding."""
    vec = embed_text(query_text)
    res = index.query(
        vector = vec,
        top_k = top_k,
        include_metadata= True,
        include_values = False
    )

    print(f"DEBUG: Pinecone RAG returned {len(res['matches'])} matches.")
    return res["matches"]

def fetch_graph_context(node_ids: List[str], neighborhood_depth=1):
    """Fetch neighboring nodes from Neo4j."""
    facts = []
    cypher_query = """
    UNWIND $node_ids AS nid
    MATCH (n:Entity {id: nid})-[r]-(m:Entity)
    RETURN n.id AS source, type(r) AS rel, m.id AS target_id, m.name AS target_name,
           m.description AS target_desc, labels(m) AS labels
    LIMIT 20
    """
    with driver.session() as session:
            recs = session.run(cypher_query, node_ids = node_ids)
            for r in recs:
                facts.append(r.data())
    print(f"DEBUG: Neo4j returned {len(facts)} graph facts.")
    return facts

def summarize_context_llm(user_query: str, pinecone_matches: List[dict], graph_facts: List[dict]) -> str:
    """Summarizes the raw context using a fast LLM (Groq)."""
    
    # Convert context lists to a simpler string format
    vec_context = "\n".join([f"- {m['id']}: {m['metadata'].get('name', '')} (Type: {m['metadata'].get('type', '')}, City: {m['metadata'].get('city', 'na')})" for m in pinecone_matches])
    graph_context = "\n".join([f"- ({f['source']}) -[{f['rel']}]-> ({f['target_id']}) {f['target_name']}" for f in graph_facts])

    prompt_content = f"""User Query: "{user_query}"

Raw Vector Matches:
{vec_context}

Raw Graph Facts:
{graph_context}

Based on the query and the raw data, provide a single, concise summary paragraph.
"""
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": LLM_SUMMARY_PROMPT},
                {"role": "user", "content": prompt_content}
            ],
            model=LLM_SUMMARIZER_MODEL,
            temperature=0.0,
            max_tokens=300
        )
        summary = chat_completion.choices[0].message.content.strip()
        return summary
    except Exception as e:
        print(f"DEBUG: Groq summary call failed ({e}), returning empty summary.")
        return "No summary could be generated."
    

def build_prompt(user_query: str, summary_context: str) -> List[dict]:
    """Build a chat prompt using the pre-summarized context"""
    system = (
        """You are an expert travel assistant for Vietnam. Your goal is to provide a helpful and accurate answer based on the user's query and the provided context.
        
        Think step-by-step:
        1. First, analyze the user's query to understand their core intent. Are they asking for an itinerary, a specific fact, a recommendation, or a comparison?
        2. Second, review the provided CONTEXT SUMMARY, which has all the relevant facts.
        3. Third, synthesize these facts into a coherent answer. 
        4. Finally, present this answer to the user in a clear and helpful format. If they asked for a plan, structure it like a plan. If they asked for a fact, state the fact directly. Always cite node ids like (attraction_123) when referencing specific places."""
    )

    prompt = [
        {"role": "system", "content": system},
        {"role": "user", "content":
         f"User query: {user_query}\n\n"
         "Here is a clean summary of all the relevant context I found:\n"
         f"{summary_context}\n\n"
         "Based on the query and the provided context summary, please create the travel plan or answer."}
    ]
    return prompt

def call_chat(prompt_messages):
    """Call OpenAI ChatCompletion."""
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=prompt_messages,
        max_tokens=600,
        temperature=0.2
    )
    return resp.choices[0].message.content

# -----------------------------
# Interactive chat
# -----------------------------
def interactive_chat():
    print("Hybrid travel assistant. Type 'exit' to quit.")
    while True:
        raw_query = input("\nEnter your travel question: ").strip()
        if not raw_query or raw_query.lower() in ("exit","quit"):
            print("Goodbye!")
            break

        # 1. Normalize the query for caching
        print("-> Normalizing query with LLM...")
        normalized_query = normalize_query_llm(raw_query)
        print(f"DEBUG: Normalized query to '{normalized_query}'")
        
        # 2. Embed the *normalized* query
        normalized_vec = embed_text(normalized_query)

        # 3. Query the SEMANTIC cache first
        try:
            cache_results = cache_index.query(
                vector=normalized_vec,
                top_k=1,
                include_metadata=True
            )
            
            if cache_results['matches']:
                top_match = cache_results['matches'][0]
                
                # 4. Check for a Cache Hit
                if top_match['score'] > CACHE_THRESHOLD:
                    print(f"\n✅ Assistant's Answer (from cache, score: {top_match['score']:.2f}):\n")
                    print(top_match['metadata']['answer']) # Retrieve saved answer
                    continue # Skip the rest and ask for the next question

        except Exception as e:
            print(f"DEBUG: Cache query failed: {e}")

        # 5. Cache Miss: Run the full RAG pipeline
        print("\n-> (Cache Miss) Retrieving new context...")
        
        # We use the RAW_QUERY for the RAG pipeline to get the best results
        matches = pinecone_query(raw_query, top_k=TOP_K)
        
        if not matches:
            print("Could not find any relevant information. Please try a different query.")
            continue
            
        match_ids = [m["id"] for m in matches]
        
        print("-> Fetching relationships from Neo4j...")
        graph_facts = fetch_graph_context(match_ids)

        print("-> Summarizing context with LLM...")
        summary = summarize_context_llm(raw_query, matches, graph_facts)
        print(f"DEBUG: Generated summary: {summary}")
        
        print("-> Building prompt and calling LLM...")
        # We pass the RAW_QUERY to the final prompt
        prompt = build_prompt(raw_query,summary )
        answer = call_chat(prompt)
        
        print("\n✅ Assistant's Answer:\n")
        print(answer)

        # 6. Save the new answer to the cache
        print("-> Saving new answer to semantic cache...")
        cache_index.upsert(
            vectors=[{
                "id": normalized_query, # Use the normalized query tag as the ID
                "values": normalized_vec,
                "metadata": {"answer": answer} # Store the full answer
            }]
        )
        print("\n=== End ===\n")

if __name__ == "__main__":
    try:
        interactive_chat()
    finally:
        driver.close()

