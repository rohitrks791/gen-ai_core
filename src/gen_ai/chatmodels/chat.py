from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chat_models import init_chat_model
from langchain_nvidia_ai_endpoints import ChatNVIDIA
# from langchain_mistralai import ChatMistralAI

# Nemotron 3 Ultra - frontier reasoning and agentic workflows
llm = ChatNVIDIA(model="nvidia/nemotron-3-ultra-550b-a55b", temperature=1)
# llm = ChatGoogleGenerativeAI(model="gemini-3.7-flash", temperature=1)

response = llm.invoke("Write a short story on AI?")
# model = init_chat_model("google_genai:gemini-3.7-flash")
# print(model)
# response = model.invoke("What is Embeddings in AI.?")
print(response.content)