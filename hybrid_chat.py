# hybrid_chat.py
import json
from typing import List
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from neo4j import GraphDatabase
import config
import re

# a simple in-memory cache
embedding_cache = {}


# -----------------------------
# Config
# -----------------------------
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
TOP_K = 5

INDEX_NAME = config.PINECONE_INDEX_NAME
CACHE_INDEX_NAME = "vietnam-travel-cache"

# -----------------------------
# Initialize clients
# -----------------------------
client = OpenAI(api_key=config.OPENAI_API_KEY)
pc = Pinecone(api_key=config.PINECONE_API_KEY)

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
# Helper functions
# -----------------------------

# A simple list of "fluff" words to ignore
ACTION_WORDS = [
    'create', 'make', 'give', 'tell', 'find', 'show', 'get', 'a', 'an', 'the', 'for', 'me',
    'i', 'would', 'like', 'to', 'go', 'on', 'can', 'you', 'provide', 'with',
    'it', 'is', 'about', 'what', 'how', 'when', 'where', 'please'
]

def normalize_query(query: str) -> str:
    """Cleans a query to focus on its core semantic topic."""
    query = query.lower()
    query = re.sub(r'[^\w\s]', '', query) # Remove punctuation

    # Remove action words
    words = query.split()
    filtered_words = [word for word in words if word not in ACTION_WORDS]

    # Sort the words to create a consistent, order-independent key
    filtered_words.sort()
    return " ".join(filtered_words)

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


def pinecone_query(query_text: str, top_k=TOP_K, query_vec = None):
    """
    Queries the Pinecone index. Can use a pre-computed vector if provided,
    otherwise it will generate one.
    """

    if query_vec is not None:
        print("DEBUG: Using pre-computed vector for Pinecone RAG query.")
        vec = query_vec
    # 2. Otherwise, generate a new embedding
    else:
        vec = embed_text(query_text)
    # vec = embed_text(query_text)
    res = index.query(
        vector=vec,
        top_k=top_k,
        include_metadata=True,
        include_values=False
    )
    print("DEBUG: Pinecone top 5 results:")
    print(len(res["matches"]))
    return res["matches"]

def fetch_graph_context(node_ids: List[str], neighborhood_depth=1):
    """Fetch neighboring nodes from Neo4j."""
    facts = []
    with driver.session() as session:
        for nid in node_ids:
            q = (
                "MATCH (n:Entity {id:$nid})-[r]-(m:Entity) "
                "RETURN type(r) AS rel, labels(m) AS labels, m.id AS id, "
                "m.name AS name, m.type AS type, m.description AS description "
                "LIMIT 10"
            )
            recs = session.run(q, nid=nid)
            for r in recs:
                facts.append({
                    "source": nid,
                    "rel": r["rel"],
                    "target_id": r["id"],
                    "target_name": r["name"],
                    "target_desc": (r["description"] or "")[:400],
                    "labels": r["labels"]
                })
    print("DEBUG: Graph facts:")
    print(len(facts))
    return facts

def build_prompt(user_query, pinecone_matches, graph_facts):
    """Build a chat prompt combining vector DB matches and graph facts."""
    system = (
        "You are a helpful travel assistant. Use the provided semantic search results "
        "and graph facts to answer the user's query briefly and concisely. "
        "Cite node ids when referencing specific places or attractions."
    )

    vec_context = []
    for m in pinecone_matches:
        meta = m["metadata"]
        score = m.get("score", None)
        snippet = f"- id: {m['id']}, name: {meta.get('name','')}, type: {meta.get('type','')}, score: {score}"
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

        # Normalize the query to get its core topic
        query = normalize_query(raw_query)
        print(f"DEBUG: Normalized query to '{query}'")

        query_vec = embed_text(query)  # This will use our dictionary cache

        try:
            cache_results = cache_index.query(
                vector = query_vec,
                top_k = 1,
                include_metadata = True
            )

            top_match = cache_results['matches'][0]

            if top_match['score'] > CACHE_THRESHOLD:
                print("\n Assistant's Answer (from cache): \n")
                print(top_match['metadata']['answer']) 
                continue
        except Exception as e:
            print("DEBUG: Cache miss or empty cache.")

        # 4. Cache Miss: Run the full pipeline as normal
        print("\n-> (Cache Miss) Retrieving new context...")
        matches = pinecone_query(query, top_k=TOP_K, query_vec=query_vec)

        if not matches:
            print("Could not find any relevant information. Please try a different query.")
            continue

        match_ids = [m["id"] for m in matches]
        graph_facts = fetch_graph_context(match_ids)
        prompt = build_prompt(query, matches, graph_facts)
        answer = call_chat(prompt)
        print("\n=== Assistant Answer ===\n")
        print(answer)

        print("-> Saving new answer to semantic cache...")
        cache_index.upsert(
            vectors=[{
                "id": query, # Use the query text as the ID
                "values": query_vec,
                "metadata": {"answer": answer} # Store the full answer
            }]
        )
        print("\n=== End ===\n")

if __name__ == "__main__":
    try:
        interactive_chat()
    finally:
        driver.close()

