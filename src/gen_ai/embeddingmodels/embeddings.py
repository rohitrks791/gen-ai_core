from dotenv import load_dotenv
load_dotenv()
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

#create embedding
embedder = NVIDIAEmbeddings(
            model="nvidia/nemotron-3-embed-1b",
            # dimensions = 64, nvidia not supports creating dimensions
        )
#For single statement
vector = embedder.embed_query("You are a GenAI expert.")

# print(vector)
# print(len(vector)) #check dimensions

texts = [
    "Hello, We are software engineers",
    "We are passionate on developing real world problems",
    "We always deliver impactful work to keep the client trust in us."
]
# to create embedding for list or document
vector_texts = embedder.embed_documents(texts)
print(vector_texts) #will have 3 embeddings inside for 3 statements
for i in vector_texts:
    print(len(i))