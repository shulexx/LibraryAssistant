from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import numpy as np
import json

app = Flask(__name__)

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

@app.route("/embed", methods=["POST"])
def embed():
    data = request.json
    text = data.get("text")
    emb = model.encode([text])[0].tolist()
    return jsonify({"embedding": emb})

@app.route("/batch-embed", methods=["POST"])
def batch_embed():
    data = request.json
    texts = data.get("texts")
    vectors = model.encode(texts).tolist()
    return jsonify({"embeddings": vectors})

if __name__ == "__main__":
    print("SBERT embedding servisi çalışıyor: http://localhost:5005")
    app.run(host="0.0.0.0", port=5005)
