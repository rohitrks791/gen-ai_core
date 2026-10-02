import os
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA

load_dotenv()

llm = ChatNVIDIA(
    model="nvidia/nemotron-3-ultra-550b-a55b",
    api_key=os.getenv("NVIDIA_API_KEY"),
)

response = llm.invoke("What is Embeddings in AI?")
print(response.content)