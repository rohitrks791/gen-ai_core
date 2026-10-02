from dotenv import load_dotenv
load_dotenv()
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
# HuggingFacePipeline can directly downloads model and make as locally availaable
hf = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 100,
        "return_full_text": False,
    },
)

chat_model = ChatHuggingFace(llm = hf)
response = chat_model.invoke("What is Generative AI?")
print(response.content)