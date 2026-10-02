from dotenv import load_dotenv

load_dotenv()

import streamlit as st

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Company Information Extractor",
    page_icon="📊",
    layout="centered"
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
You are an expert information extraction and summarization assistant.

Your task is to analyze the company-related text provided by the user
and extract useful, factual, and relevant information from it.

Extract the following information:

1. Company Name
2. Industry
3. Company Overview
4. Products / Services
5. Business Segments
6. Financial Information
7. Revenue / Earnings
8. Growth Metrics
9. AI Involvement
10. Cloud / Technology Involvement
11. Key Strengths
12. Other Important Information
13. Quick Summary

Rules:
- Extract information only from the provided text.
- Do not assume, infer, or add information that is not explicitly mentioned.
- If any information is not available, write "Not mentioned".
- Preserve important numbers, percentages, dates, currencies, and financial figures.
- Keep the extracted information concise and easy to understand.
- List multiple products, services, or segments as bullet points.
- The Quick Summary must be 2-3 sentences maximum.
- Focus on information that provides useful insight into the company's
  business, products, financial performance, AI involvement, and technology.

Return the answer in a clean and structured format.
"""
    ),
    (
        "human",
        """
Analyze the following company paragraph and extract the required information:

{company_text}
"""
    )
])


# --------------------------------------------------
# UI
# --------------------------------------------------

st.title("📊 Company Information Extractor")
st.write("Enter a company-related paragraph to extract useful information.")


company_text = st.text_area(
    "Company Text",
    placeholder="Enter the company paragraph here...",
    height=250
)


if st.button("Extract Information"):

    if company_text:

        messages = prompt_template.invoke({
            "company_text": company_text
        })

        response = model.invoke(messages)

        st.subheader("Extracted Information")

        st.write(response.content)

    else:
        st.warning("Please enter company text.")

#Right now your app prints nice text. But companies dont want text, They want data they can store, search, filter, reccommend, analyze, send to APIs.This is called Structured Output
# AI -> JSON -> Backend -> API -> Frontend, Without structured output AI breaks the system every time.
# To generate proper structured output or proper object, we need to use Pydantic for that case.