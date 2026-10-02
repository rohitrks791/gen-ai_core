from dotenv import load_dotenv
load_dotenv()

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser

#creating schema
class Company(BaseModel):
    company_name: str
    established_year: int
    industry: str
    company_overview: str
    products_services: List[str]
    financial_information: str
    ai_involvement: str
    key_strength: str
    financial_earnings: str
    summary: str
    rating: Optional[float]

# To verify the filled data, there is pydantic parser to use
parser = PydanticOutputParser(pydantic_object=Company)
model = ChatNVIDIA(
            model="nvidia/nemotron-3-super-120b-a12b",
            temperature=0.6
        )

prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        """
Extract company information from the paragraph
    {format_instruction}
"""
    ),
    (
        "human",   "{company_text}"
    )
])

company_text = input("Give your company text: ")

messages = prompt_template.invoke({
    "company_text": company_text,
    "format_instruction": parser.get_format_instructions()
})

response = model.invoke(messages) #we will json object notation bcz of pydantic schema, and it is following schema is checked by parser
response_data = parser.parse(response.content)

print(response_data)