import os
   
from phi.tools.firecrawl import FirecrawlTools
from phi.tools.duckduckgo import DuckDuckGo
from phi.model.openai import OpenAIChat
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.embedder.openai import OpenAIEmbedder
from phi.vectordb.lancedb import LanceDb, SearchType
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import helium
from typing import List
from pydantic import BaseModel, Field
from fastapi import UploadFile

# Load environment variables (API keys, etc.)
from dotenv import load_dotenv
load_dotenv()

##############################
# 1️⃣ Company Search Agent   #
##############################
company_search_agent = Agent(
    name="Company Search Agent",
    model=OpenAIChat(id="gpt-4o"),
    tools=[DuckDuckGo()],
    description="Finds company details based on name using web search.",
    instructions=["Always include sources in search results."],
    show_tool_calls=True,
    markdown=True,
)

def search_company(company_name: str):
    query = f"Find detailed company information for {company_name}. Extract its official website, mission, services, and any AI-related initiatives. Prioritize official sources and provide links where available."
    return company_search_agent.print_response(query)


##############################
# 2️⃣ Website Scraper Agent   #
##############################
firecrawl_agent = Agent(
    name="Website Scraper Agent",
    tools=[FirecrawlTools(scrape=True, crawl=False)],
    description="Extracts content from company websites.",
    show_tool_calls=True,
    markdown=True,
)

def scrape_website(url: str):
    return firecrawl_agent.print_response(f"Extract all relevant business information from {url}, including mission statement, services, case studies, and AI-related content. Provide structured output.")

# Helium for dynamic websites
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--headless")
driver = helium.start_chrome(headless=True, options=chrome_options)

def scrape_dynamic_website(url: str):
    helium.go_to(url)
    text = helium.get_driver().page_source
    return text


##############################
# 3️⃣ Text Processing Agent   #
##############################
class CompanySummary(BaseModel):
    summary: str = Field(..., description="Summarized company details based on user input.")

text_processing_agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    description="Summarizes user-written company descriptions.",
    response_model=CompanySummary,
)

def process_company_description(text: str):
    return text_processing_agent.print_response(f"Summarize the following company description: {text}. Focus on key services, mission, industry, and potential AI use cases where applicable.")


#################################
# 4️⃣ Document Processing Agent  #
#################################
# LanceDB for storing extracted knowledge
knowledge_base = PDFUrlKnowledgeBase(
    urls=[],  # PDFs will be dynamically added
    vector_db=LanceDb(
        table_name="company_docs",
        uri="tmp/lancedb",
        search_type=SearchType.vector,
        embedder=OpenAIEmbedder(model="text-embedding-3-small"),
    ),
)
knowledge_base.load(recreate=False)

document_processing_agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    knowledge=knowledge_base,
    description="Extracts and processes data from uploaded PDFs/PPTs.",
    show_tool_calls=True,
    markdown=True,
)

def process_uploaded_document(file: UploadFile):
    file_path = f"tmp/{file.filename}"
    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())
    
    knowledge_base.load(recreate=False)
    return document_processing_agent.print_response(f"Analyze and extract key insights from the uploaded document: {file.filename}. Summarize business operations, AI-related discussions, financial details, and relevant strategic insights.")


###########################
# Example Usage           #
###########################
if __name__ == "__main__":
    company_name = "Tesla"
    print("Company Search Results:")
    search_company(company_name)
    
    website_url = "https://www.tesla.com"
    print("\nScraped Website Data:")
    scrape_website(website_url)
    
    user_description = "We are a renewable energy startup focusing on solar solutions."
    print("\nProcessed Company Description:")
    process_company_description(user_description)
    
    # Example of handling an uploaded file
    # process_uploaded_document(uploaded_file)
