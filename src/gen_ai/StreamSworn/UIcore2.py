from dotenv import load_dotenv

load_dotenv()

import streamlit as st

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Company Information Extractor",
    page_icon="🏢",
    layout="centered"
)


# --------------------------------------------------
# Creating Schema
# --------------------------------------------------

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


# --------------------------------------------------
# Pydantic Output Parser
# --------------------------------------------------

parser = PydanticOutputParser(
    pydantic_object=Company
)


# --------------------------------------------------
# Model
# --------------------------------------------------

model = ChatNVIDIA(
    model="nvidia/nemotron-3-super-120b-a12b",
    temperature=0.6
)


# --------------------------------------------------
# Prompt Template
# --------------------------------------------------

prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        """
Extract company information from the paragraph.

{format_instruction}
"""
    ),
    (
        "human",
        "{company_text}"
    )
])


# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------

st.title("🏢 Company Information Extractor")

st.write(
    "Enter a company-related paragraph to extract structured information."
)


company_text = st.text_area(
    "Company Text",
    placeholder="Enter the company paragraph here...",
    height=250
)


# --------------------------------------------------
# Extract Information
# --------------------------------------------------

if st.button("Extract Information"):

    if company_text:

        messages = prompt_template.invoke({
            "company_text": company_text,
            "format_instruction": parser.get_format_instructions()
        })

        response = model.invoke(messages)

        # Parse response using Pydantic schema
        response_data = parser.parse(response.content)

        st.subheader("📋 Extracted Company Information")

        st.write(response_data)

    else:

        st.warning("Please enter company text.")