from dotenv import load_dotenv
load_dotenv()

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

model = ChatNVIDIA(
            model="nvidia/nemotron-3-super-120b-a12b",
            temperature=0.6
        )

#If you are using basic LLM then we can use PromptTemplate(from_template),otherwise if its a AI chatbot itself, then its recommend to use ChatPromptTemplate(from_messages)
# prompt_template = PromptTemplate(
#     input_variables=["text"],
#     template="""
#         You are an information extraction and summarization expert.

#         Analyze the following company-related paragraph and extract the most useful
#         and relevant information from it.

#         Return the information in the following structured format:

#         Company Name:
#         Industry:
#         Company Overview:
#         Products / Services:
#         Business Segments:
#         Financial Information:
#         Growth Metrics:
#         AI Involvement:
#         Cloud / Technology Involvement:
#         Key Strengths:
#         Other Important Information:
#         Quick Summary:

#         Rules:
#         - Extract information only from the given paragraph.
#         - Do not make assumptions or add information that is not present.
#         - If a field is not mentioned, write "Not mentioned".
#         - Keep extracted information concise and easy to understand.
#         - For Products / Services and Business Segments, use bullet points.
#         - Preserve important numbers, percentages, years, and financial figures.
#         - The Quick Summary should be 2-3 sentences maximum.
#         - Focus on information that would be useful for understanding the company,
#         its business, technology, AI involvement, and financial performance.

#         Company paragraph:
#         {text}
#     """
# )

# text = """
# Microsoft is a technology company focused on empowering people and
# organizations through digital tools and AI. Its main businesses are
# Productivity (Microsoft 365, LinkedIn), Intelligent Cloud (Azure), and
# Personal Computing (Windows, Xbox). In FY2026, it earned $285.1 billion in
# revenue, with Azure growing 41%, making it a top leader in cloud and AI.
# """

# use chatgpt to create the prompt to use in PromptTemplate
# prompt = prompt_template.invoke({"text": text})
# response = model.invoke(prompt)

# print(response.content) 
#Prompt template: A template is a kind of structured prompt that we can reuse it again n again when we want to use it for specific use-case

# Instantiation using from_template (recommended)
# prompt = PromptTemplate.from_template("Say {foo}")
# prompt.format(foo="bar")

# Instantiation using initializer
# prompt = PromptTemplate(template="Say {foo}")
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

# company_text = """
# Microsoft is a technology company focused on empowering people and
# organizations through digital tools and AI. Its main businesses are
# Productivity (Microsoft 365, LinkedIn), Intelligent Cloud (Azure), and
# Personal Computing (Windows, Xbox). In FY2026, it earned $285.1 billion in
# revenue, with Azure growing 41%, making it a top leader in cloud and AI.
# """

company_text = input("Give your company text: ")

messages = prompt_template.invoke({
    "company_text": company_text
})

response = model.invoke(messages)

print(response.content)