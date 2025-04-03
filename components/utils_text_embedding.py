import requests

def get_text_embedding(texts, instruction=None, server_url="http://localhost:5000"):
    """
    Get embeddings for a list of texts from the embedding server.
    
    Args:
        texts (list): List of strings to embed
        instruction (str, optional): Instruction for the embedding model
        server_url (str): Base URL of the embedding server
        
    Returns:
        list: List of embeddings
    """
    payload = {
        "texts": texts,
        "instruction": instruction or ""
    }
    try:
        response = requests.post(f"{server_url}/encode", json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()["embeddings"]
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to embedding server: {e}")
        return None

def test_embedding_client():
    test_texts = [
        "Hello, world!",
        "This is a test."
    ]
    
    embeddings = get_text_embedding(test_texts)
    if embeddings:
        print("Successfully got embeddings!")
        print(f"Number of embeddings: {len(embeddings)}")
        print(f"Embedding dimension: {len(embeddings[0])}")
    else:
        print("Failed to get embeddings")

if __name__ == "__main__":
    test_embedding_client() 