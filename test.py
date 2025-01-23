import json
import numpy as np

# Load small dataset
training_data = {
    "hello": "Hi there!",
    "how are you": "I'm doing well, thanks!",
    "what is your name": "I am a simple model.",
    "tell me a joke": "Why did the chicken cross the road? To get to the other side!"
}

# Basic token embedding using ASCII values
def embed_text(text):
    return np.array([ord(c) for c in text.lower()]) / 255  # Normalize ASCII values


# Compute similarity using cosine similarity
def cosine_similarity(vec1, vec2):
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Simple "attention mechanism" to find the best matching response
def simple_llm(query):
    query_vec = embed_text(query)
    
    best_match = None
    best_score = -1

    for key in training_data:
        key_vec = embed_text(key)
        score = cosine_similarity(query_vec, key_vec)
        
        if score > best_score:
            best_score = score
            best_match = key
    
    return training_data.get(best_match, "I don't understand.")

# Example usage
print(simple_llm("hello"))  # Expected: "Hi there!"
print(simple_llm("who are you"))  # Expected: "I am a simple model."
print(simple_llm("tell joke"))  # Expected: "Why did the chicken cross the road?"
