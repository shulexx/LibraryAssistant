from sentence_transformers import SentenceTransformer
import json

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

with open("C:/Users/Lenovo/Desktop/MK/AI.Library.Service/AI.Library.Service/Data/books.json", "r", encoding="utf-8-sig") as f:
    books = json.load(f)

texts = [
    f"{b['title']} {b['author']} {b['subject']} {b['summary']} {' '.join(b['keywords'])}"
    for b in books
]

vectors = model.encode(texts).tolist()

out = []
for b, emb in zip(books, vectors):
    out.append({"isbn": b["isbn"], "embedding": emb})

with open("path/AI.Library.Service/AI.Library.Service/Data/embeddings.json", "w", encoding="utf-8-sig") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

print("embeddings.json oluşturuldu ✔")
