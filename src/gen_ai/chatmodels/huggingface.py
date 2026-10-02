from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4.1-Flash"
)

model = ChatHuggingFace(llm=llm)
response = model.invoke("Who are you?")
print(response.content)