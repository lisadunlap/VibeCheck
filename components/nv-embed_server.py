from flask import Flask, request, jsonify
from transformers import AutoModel
import torch


app = Flask(__name__)

model = AutoModel.from_pretrained(
            'nvidia/NV-Embed-v2',
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="balanced_low_0",
            low_cpu_mem_usage=True
        )

@app.route('/encode', methods=['POST'])
def encode_texts():
    """
    Endpoint to encode texts into embeddings.
    Expects a JSON payload with 'texts' and optional 'instruction'.
    """
    data = request.json
    texts = data.get('texts', [])
    instruction = data.get('instruction', "")
    
    if not texts:
        return jsonify({"error": "No texts provided"}), 400

    try:
        embeddings = model.encode(texts, instruction=instruction)
        embeddings_list = embeddings.cpu().numpy().tolist()
        return jsonify({"embeddings": embeddings_list})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/retrieve', methods=['POST'])
def retrieve():
    """
    Endpoint to retrieve similar passages.
    Expects a JSON payload with 'query_texts', 'top_k', and optional 'instruction'.
    """
    data = request.json
    query_texts = data.get('query_texts', [])
    top_k = data.get('top_k', 3)
    instruction = data.get('instruction', None)
    
    if not query_texts:
        return jsonify({"error": "No query texts provided"}), 400

    try:
        embeddings = model.encode(query_texts, instruction=instruction)
        embeddings_list = embeddings.cpu().numpy().tolist()
        return jsonify({"embeddings": embeddings_list})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
def test_embedding_server():
    import requests
    import json
    print("Testing embedding server...")

    test_data = {
        "texts": ["Hello, world!", "This is a test."],
        "instruction": "Represent the following text in a way that is suitable for semantic search."
    }

    response = requests.post('http://localhost:5000/encode', json=test_data)
    print(response.json())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    test_embedding_server()