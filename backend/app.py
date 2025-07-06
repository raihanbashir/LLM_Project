from flask import Flask, request, jsonify
import requests
import os
import json
import time
from datetime import datetime
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import numpy as np
import hashlib

# Initialize Sentence Transformer Model
model = SentenceTransformer('all-MiniLM-L6-v2')

load_dotenv()

app = Flask(__name__)

# LM Studio API configuration
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
LM_MODEL = "lmstudio-community/Llama-3.2-1B-Instruct-GGUF"

# AviationStack API Key
AVIATIONSTACK_API_KEY = os.getenv('AVIATIONSTACK_API_KEY')

def get_cached_info(destination):
    """Check if we have cached information for this destination"""
    cache_dir = "travel_cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create a hash of the destination for the filename
    cache_key = hashlib.md5(destination.lower().encode()).hexdigest()
    cache_file = os.path.join(cache_dir, f"{cache_key}.json")
    
    if os.path.exists(cache_file):
        # Check if cache is still fresh (7 days)
        cache_time = os.path.getmtime(cache_file)
        if (time.time() - cache_time) < (7 * 24 * 60 * 60):  # 7 days in seconds
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
    return None

def save_to_cache(destination, data):
    """Save destination information to cache"""
    cache_dir = "travel_cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_key = hashlib.md5(destination.lower().encode()).hexdigest()
    cache_file = os.path.join(cache_dir, f"{cache_key}.json")
    
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump({"data": data, "timestamp": datetime.now().isoformat()}, f)

def fetch_and_clean_text(url):
    """Fetches and cleans text content from a URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'lxml')
        for script_or_style in soup(['script', 'style', 'nav', 'footer', 'header']):
            script_or_style.decompose()
        # Split by paragraphs and join, to have some structure
        paragraphs = [p.get_text() for p in soup.find_all('p')]
        return '\n'.join(p for p in paragraphs if len(p) > 20) # filter short/empty paragraphs
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return ""

def get_relevant_info(destination, interests):
    """Fetch relevant information using web scraping and sentence transformers."""
    cached = get_cached_info(destination)
    if cached:
        print(f"Returning cached info for {destination}.")
        return cached.get("data", "")

    print(f"Fetching new info for {destination} from the web.")
    urls = [
        f"https://en.wikipedia.org/wiki/{destination.replace(' ', '_')}",
        f"https://wikitravel.org/en/{destination.replace(' ', '_')}"
    ]
    
    all_chunks = []
    for url in urls:
        text = fetch_and_clean_text(url)
        if text:
            # Split text into chunks (paragraphs in this case)
            chunks = [p.strip() for p in text.split('\n') if len(p.strip()) > 100]
            all_chunks.extend(chunks)
    
    if not all_chunks:
        print("No content fetched or processed.")
        return ""

    # Generate embeddings for text chunks
    chunk_embeddings = model.encode(all_chunks)
    
    # Create query and generate its embedding
    query = f"Travel guide for {destination} focusing on {interests}"
    query_embedding = model.encode([query])
    
    # Find the most relevant chunks using cosine similarity
    # Using numpy for dot product
    similarities = np.dot(chunk_embeddings, query_embedding.T).flatten()
    
    top_k = min(5, len(all_chunks)) # Get top 5 or fewer
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    relevant_context = "\n\n".join([all_chunks[i] for i in top_indices])
    
    print(f"Found relevant context, caching {len(relevant_context)} characters.")
    save_to_cache(destination, relevant_context)
    
    return relevant_context

# Itinerary endpoint with RAG
@app.route('/api/itinerary', methods=['POST'])
def generate_itinerary():
    data = request.json
    destination = data.get('destination')
    duration = data.get('duration')
    interests = data.get('interests')
    
    # Get relevant information using RAG
    context = get_relevant_info(destination, interests)
    
    prompt = f"""
    Generate a detailed {duration}-day itinerary for {destination} focusing on {interests}.
    
    Here's some relevant information to help you create an accurate itinerary:
    {context}
    
    The itinerary should include:
    1. A brief introduction to the destination
    2. Daily schedule with specific times (morning, afternoon, evening)
    3. Specific activities and attractions
    4. Local tips and recommendations
    5. Estimated time needed for each activity
    
    If the information above is not relevant, use your general knowledge to create 
    a well-structured and realistic itinerary.
    """
    
    return query_llm(prompt)

# Budget calculator endpoint
@app.route('/api/budget', methods=['POST'])
def calculate_budget():
    data = request.json
    destination = data.get('destination')
    duration = data.get('duration')
    budget = data.get('budget')
    
    prompt = f"""
    Create a detailed budget breakdown for a {duration}-day trip to {destination}.
    Total budget: ${budget}
    Break down into categories: accommodation, food, transportation, activities, and miscellaneous.
    Provide average daily costs and total for each category.
    """
    
    return query_llm(prompt)

# Flight information endpoint
@app.route('/api/flights', methods=['POST'])
def get_flight_info():
    data = request.json
    origin = data.get('origin')
    destination = data.get('destination')
    departure_date = data.get('departure_date')
    
    try:
        # AviationStack expects YYYY-MM-DD
        formatted_date = departure_date  # already in correct format from frontend

        aviationstack_url = "http://api.aviationstack.com/v1/flights"
        params = {
            'access_key': AVIATIONSTACK_API_KEY,
            'dep_iata': origin,
            'arr_iata': destination,
            'flight_date': formatted_date,
            'limit': 10
        }
        response = requests.get(aviationstack_url, params=params)
        flights_data = response.json()

        if not flights_data.get('data'):
            return jsonify({"error": "No flights found"}), 404
        flights = flights_data['data']

        results = []
        for f in flights:
            flight_info = {
                'airline': f.get('airline', {}),
                'flight': f.get('flight', {}),
                'departure': f.get('departure', {}),
                'arrival': f.get('arrival', {}),
                'flight_status': f.get('flight_status'),
                'flight_date': formatted_date,
                'status': f.get('flight_status', 'scheduled').lower()
            }
            # Clean up the data structure
            if 'scheduled' in flight_info['departure']:
                flight_info['departure']['scheduled'] = flight_info['departure']['scheduled'].replace('+00:00', 'Z')
            if 'scheduled' in flight_info['arrival']:
                flight_info['arrival']['scheduled'] = flight_info['arrival']['scheduled'].replace('+00:00', 'Z')
            results.append(flight_info)
        return jsonify({"flights": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Helper function for LLM queries
def query_llm(prompt):
    try:
        response = requests.post(
            LM_STUDIO_URL,
            json={
                "model": LM_MODEL,
                "messages": [
                    {"role": "system", "content": "You are a knowledgeable travel assistant that creates detailed and realistic travel itineraries. Provide specific recommendations and practical information."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 2000
            },
            headers={"Content-Type": "application/json"},
            timeout=120
        )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error in query_llm: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
