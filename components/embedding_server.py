from flask import Flask, request, jsonify
import torch
from InstructorEmbedding import INSTRUCTOR
from sentence_transformers import SentenceTransformer
from typing import List


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # # Initialize the device and model
# model = INSTRUCTOR('hkunlp/instructor-xl').to(device)

# Initialize Flask app
app = Flask(__name__)

query_prompt_name = "s2s_query"
model = SentenceTransformer(
    "dunzhang/stella_en_1.5B_v5",
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="balanced_low_0",
    low_cpu_mem_usage=True,
).to(device)


def get_embedding(instruction: str, texts: List[str]):
    if instruction != "None":
        embeddings = model.encode(texts, prompt_name=query_prompt_name)
    else:
        embeddings = model.encode(texts)
    # Convert tensor to list for JSON response
    return embeddings.tolist()


# # Function to get the embedding
# def get_embedding(instruction, texts):
#     inputs = [[instruction, text] for text in texts]
#     print("INPUTS", inputs)
#     embeddings = model.encode(inputs, convert_to_tensor=True)
#     # Convert tensor to list for JSON response
#     return embeddings.cpu().numpy().tolist()


# Define the /get_embedding endpoint
@app.route("/get_embedding", methods=["POST"])
def handle_embedding_request():
    data = request.get_json()
    instruction = data.get("instruction")
    text = data.get("text")
    print("instruction", instruction)
    print("text", text)

    if not instruction or not text:
        return jsonify({"error": "Both instruction and text must be provided"}), 400

    # try:
    embedding = get_embedding(instruction, text)
    print("embedding", embedding)
    return jsonify({"embedding": embedding}), 200
    # except Exception as e:
    #     return jsonify({'error': str(e)}), 500


# Run the server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)
    # test with curl -X POST http://localhost:5000/get_embedding -H "Content-Type: application/json" -d '{"instruction": "Summarize the following text", "text": "This is an example text."}'
