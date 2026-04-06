import faiss
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer
from google import genai  # ✅ Updated import

# ✅ Initialize modern Gemini Client
# The 'gemini-3.1-flash-lite-preview' model is the active version for the v1beta endpoint
client = genai.Client(api_key="AWRG64")
MODEL_ID = "gemini-3.1-flash-lite-preview"

# ✅ Load embedding model (FREE)
# Silencing the parallelism warning for cleaner logs
os.environ["TOKENIZERS_PARALLELISM"] = "false"
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

def ask_question(question):
    if not os.path.exists("vectors.index") or not os.path.exists("chunks.pkl"):
        print("❌ Vector database not found!")
        return None

    try:
        # Load FAISS index and chunks
        index = faiss.read_index("vectors.index")
        with open("chunks.pkl", "rb") as f:
            data = pickle.load(f)

        chunks = data['chunks'] if isinstance(data, dict) else data

        # ✅ Embed question
        query_vector = embed_model.encode([question])
        query_vector = np.array(query_vector).astype('float32')

        # Search top 3 matches
        scores, indices = index.search(query_vector, 3)

        print("🔍 Top matches:")
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            print(f"   {i+1}. Score: {score:.4f}")

        # Build context from retrieved chunks
        context = "\n\n".join([chunks[idx] for idx in indices[0]])

        # ✅ Gemini 3.1 Generation
        prompt = f"""
        You are a helpful assistant. Answer ONLY using the provided context.
        If the answer is not in the context, say "I don't know based on the provided documents."

        Context:
        {context}

        Question:
        {question}
        """

        # New SDK syntax: client.models.generate_content
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt
        )

        return response.text

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None

def main():
    if not os.path.exists("vectors.index") or not os.path.exists("chunks.pkl"):
        print("❌ Run your PDF indexing script first!")
        return

    print("\n" + "=" * 50)
    print("🤖 RAG System (Gemini 3.1 + FREE embeddings)")
    print("Type 'exit' to quit")
    print("=" * 50)

    while True:
        try:
            question = input("\n❓ Question: ").strip()
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break

        if question.lower() in ['exit', 'quit', 'q']:
            print("👋 Goodbye!")
            break

        if not question:
            continue

        print("🔍 Searching + Generating answer...")
        answer = ask_question(question)

        if answer:
            print(f"\n🤖 Answer:\n{answer}")
        else:
            print("❌ Failed to generate answer.")

if __name__ == "__main__":
    main()