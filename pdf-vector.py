import faiss
import PyPDF2
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

# FREE embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')


def pdf_to_vectors(pdf_path):
    print(f"📄 Reading PDF: {pdf_path}")

    with open(pdf_path, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        total_pages = len(pdf_reader.pages)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""

    print(f"📊 Total pages: {total_pages}")
    print(f"📊 Total text length: {len(text)} characters")

    # Chunking
    chunks = []
    for i in range(0, len(text), 400):
        chunks.append(text[i:i + 500])

    print(f"✂️ Created {len(chunks)} chunks")

    # ✅ FREE embeddings (NO OpenAI)
    print("🔄 Generating embeddings locally...")
    embeddings = model.encode(chunks)

    embeddings = np.array(embeddings).astype('float32')

    # FAISS index (384 dims)
    print("🗂️ Creating FAISS index...")
    index = faiss.IndexFlatIP(384)
    index.add(embeddings)

    # Save
    faiss.write_index(index, "vectors.index")

    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print("✅ Done! Files saved: vectors.index, chunks.pkl")


if __name__ == "__main__":
    pdf_file = "C:/Users/IT'S MY/Downloads/resume__Copy_ (4).pdf"
    pdf_to_vectors(pdf_file)