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

LLM_SYSTEM_PROMPT = """You are an ultra-fast query tagging engine. Your sole purpose is to analyze a user's travel query and extract three specific entities: LOCATION, DURATION, and TOPIC.

You MUST follow these rules:
1.  **LOCATION:** The primary city or country. If not specified, use 'na'.
2.  **DURATION:** The time (e.g., '2 day', '1 week'). If not specified, use 'na'.
3.  **TOPIC:** The main intent (e.g., 'itinerary', 'sightings', 'food', 'hotel', 'romantic', 'adventure', 'beach'). If it's a general request, use 'general'.
4.  **FORMAT:** You MUST reply with ONLY a single line in the format: `LOCATION | DURATION | TOPIC`.
5.  **DO NOT** add any explanation, preamble, or apologies.
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
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
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

    # print("DEBUG: Pinecone top 5 results:")
    # print(len(res["matches"]))
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
        # for nid in node_ids:
            # q = (
            #     "MATCH (n:Entity {id:$nid})-[r]-(m:Entity) "
            #     "RETURN type(r) AS rel, labels(m) AS labels, m.id AS id, "
            #     "m.name AS name, m.type AS type, m.description AS description "
            #     "LIMIT 10"
            # )
            recs = session.run(cypher_query, node_ids = node_ids)
            for r in recs:
                facts.append({
                    "source": r["source"],
                    "rel": r["rel"],
                    "target_id": r["target_id"],
                    "target_name": r["target_name"],
                    "target_desc": (r["target_desc"] or "")[:400],
                    "labels": r["labels"]
                })
    # print("DEBUG: Graph facts:")
    # print(len(facts))
    print(f"DEBUG: Neo4j returned {len(facts)} graph facts.")
    return facts

def build_prompt(user_query, pinecone_matches, graph_facts):
    """Build a chat prompt combining vector DB matches and graph facts."""
    system = (
        """You are an expert travel assistant for Vietnam. Your goal is to provide a helpful and accurate answer based on the user's query and the provided context.
        
        Think step-by-step:
        1. First, analyze the user's query to understand their core intent. Are they asking for an itinerary, a specific fact, a recommendation, or a comparison?
        2. Second, review the provided semantic matches and graph facts to find the information that is most relevant to the user's specific intent.
        3. Third, synthesize the relevant information into a coherent answer. 
        4. Finally, present this answer to the user in a clear and helpful format. If they asked for a plan, structure it like a plan. If they asked for a fact, state the fact directly. Always cite node ids like (attraction_123) when referencing specific places."""
    )

    vec_context = []
    for m in pinecone_matches:
        meta = m["metadata"]
        score = m.get("score", 0)
        snippet = f"- id: {m['id']}, name: {meta.get('name','')}, type: {meta.get('type','')}, score: {score:.2f}"
        if meta.get("city"):
            snippet += f", city: {meta.get('city')}"
        vec_context.append(snippet)

    graph_context = [
        f"- ({f['source']}) -[{f['rel']}]-> ({f['target_id']}) {f['target_name']}: {f['target_desc']}"
        for f in graph_facts
    ]

    prompt = [
        {"role": "system", "content": system},
        {"role": "user", "content":
         f"User query: {user_query}\n\n"
         "Top semantic matches (from vector DB):\n" + "\n".join(vec_context[:10]) + "\n\n"
         "Graph facts (neighboring relations):\n" + "\n".join(graph_context[:20]) + "\n\n"
         "Based on the above, answer the user's question. If helpful, suggest 2–3 concrete itinerary steps or tips and mention node ids for references."}
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
        
        print("-> Building prompt and calling LLM...")
        # We pass the RAW_QUERY to the final prompt
        prompt = build_prompt(raw_query, matches, graph_facts)
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

