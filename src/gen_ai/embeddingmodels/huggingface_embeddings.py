from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

texts = [
    "Hello, we are software engineers.",
    "We are passionate about developing real-world solutions.",
    "We always deliver impactful work to maintain client trust."
]

vectors = embeddings.embed_documents(texts)

print("Number of vectors:", len(vectors))
print("Dimensions:", len(vectors[0]))
print("First vector:", vectors[0])