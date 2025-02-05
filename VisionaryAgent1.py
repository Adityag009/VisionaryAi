import os
from phi.agent import Agent
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
from selenium import webdriver
import helium
from selenium.webdriver.chrome.service import Service

# Set paths explicitly
CHROME_PATH = "/usr/bin/chromium-browser"
CHROMEDRIVER_PATH = "/usr/local/bin/chromedriver"

# Configure Chrome options for Hugging Face Spaces
chrome_options = webdriver.ChromeOptions()
chrome_options.binary_location = CHROME_PATH  # Manually specify Chromium binary
chrome_options.add_argument("--headless")  # Run in headless mode
chrome_options.add_argument("--no-sandbox")  # Required for Hugging Face Spaces
chrome_options.add_argument("--disable-dev-shm-usage")  # Prevents memory issues
chrome_options.add_argument("--disable-gpu")  # Disable GPU acceleration
chrome_options.add_argument("--remote-debugging-port=9222")  # Helps debugging

# Initialize Chrome WebDriver with the correct service path
service = Service(CHROMEDRIVER_PATH)
driver = webdriver.Chrome(service=service, options=chrome_options)

# Start Helium using the modified driver
helium.set_driver(driver)


# Load environment variables (API keys, etc.)
from dotenv import load_dotenv
load_dotenv()

#####################################################################################
#                                    PHASE 1                                        #
#####################################################################################


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
# chrome_options = webdriver.ChromeOptions()
# chrome_options.add_argument("--headless")
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


#####################################################################################
#                                    PHASE 2                                        #
#####################################################################################


###########################
# Example Usage           #
###########################
# if __name__ == "__main__":
    # company_name = "Tesla"
    # print("Company Search Results:")
    # search_company(company_name)
    
    # website_url = "https://www.tesla.com"
    # print("\nScraped Website Data:")
    # scrape_website(website_url)
    
    # user_description = "We are a renewable energy startup focusing on solar solutions."
    # print("\nProcessed Company Description:")
    # process_company_description(user_description)
    
    # Example of handling an uploaded file
    # process_uploaded_document(uploaded_file)


    import os
from phi.agent import Agent
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.exa import ExaTools
from phi.model.openai import OpenAIChat
from typing import List
from pydantic import BaseModel, Field

# Load environment variables (API keys, etc.)
from dotenv import load_dotenv
load_dotenv()

##############################
# 1️⃣ Industry Trends Agent  #
##############################
industry_trends_agent = Agent(
    name="Industry Trends Agent",
    model=OpenAIChat(id="gpt-4o"),
    tools=[ExaTools(include_domains=["cnbc.com", "reuters.com", "bloomberg.com"])],
    description="Finds the latest AI advancements in a given industry.",
    show_tool_calls=True,
    markdown=True,
)

def get_industry_trends(industry: str):
    query = f"Find the latest AI advancements, innovations, and emerging technologies in the {industry} sector. Include breakthroughs, adoption trends, and notable implementations by leading companies. Provide references and insights from credible sources."
    return industry_trends_agent.print_response(query)


##################################
# 2️⃣ AI Use Case Discovery Agent #
##################################
ai_use_case_agent = Agent(
    name="AI Use Case Discovery Agent",
    model=OpenAIChat(id="gpt-4o"),
    tools=[DuckDuckGo()],
    description="Identifies AI applications relevant to a given industry.",
    show_tool_calls=True,
    markdown=True,
)

def get_ai_use_cases(industry: str):
    query = f"Identify the most impactful AI use cases in the {industry} sector. Include real-world applications, automation improvements, cost-saving innovations, and data-driven decision-making processes. Provide case studies and examples of successful AI implementation."
    return ai_use_case_agent.print_response(query)


####################################
# 3️⃣ Competitive Analysis Agent   #
####################################
competitive_analysis_agent = Agent(
    name="Competitive Analysis Agent",
    model=OpenAIChat(id="gpt-4o"),
    tools=[DuckDuckGo(), ExaTools(include_domains=["techcrunch.com", "forbes.com", "businessinsider.com"])],
    description="Analyzes how competitors are using AI in their businesses.",
    show_tool_calls=True,
    markdown=True,
)

def get_competitor_ai_strategies(company_name: str):
    query = f"Analyze how {company_name} is leveraging AI in its business operations. Find recent reports, product innovations, automation strategies, and AI-driven transformations. Highlight competitive advantages gained through AI adoption. Provide references and sources."
    return competitive_analysis_agent.print_response(query)


###########################
# Example Usage           #
###########################
# if __name__ == "__main__":
#     industry = "Healthcare"
#     print("Industry Trends:")
#     get_industry_trends(industry)
    
#     print("\nAI Use Cases:")
#     get_ai_use_cases(industry)
    
#     competitor = "Pfizer"
#     print("\nCompetitor AI Strategies:")
#     get_competitor_ai_strategies(competitor)


#####################################################################################
#                                    PHASE 3                                        #
#####################################################################################


##############################
# 1️⃣ Reasoning Agent        #
##############################
reasoning_agent = Agent(
    name="Reasoning Agent",
    model=OpenAIChat(id="gpt-4o"),
    description="Processes all collected data and generates structured AI adoption strategies.",
    show_tool_calls=True,
    markdown=True,
)

def generate_ai_strategy(company_data: str, industry_trends: str, ai_use_cases: str, competitor_analysis: str):
    query = f"""
    You are an AI business strategist analyzing a company's potential AI adoption. Given the following:
    
    - **Company Overview:** {company_data}
    - **Industry Trends:** {industry_trends}
    - **AI Use Cases:** {ai_use_cases}
    - **Competitor AI Strategies:** {competitor_analysis}
    
    Generate a structured AI adoption strategy that includes:
    1. **AI Opportunities**: Identify key areas where AI can enhance operations, customer experience, or business efficiency.
    2. **Technology Fit**: Recommend specific AI tools, models, or methodologies that fit this company's needs.
    3. **Implementation Roadmap**: Step-by-step guidance on integrating AI, considering costs, scalability, and ROI.
    4. **Future Scalability**: How AI adoption can evolve over time for long-term growth.
    
    Provide structured insights with a logical flow and avoid generic statements. Use industry benchmarks where possible.
    """
    return reasoning_agent.print_response(query)


##############################
# 2️⃣ AI Integration Advisor  #
##############################
ai_integration_agent = Agent(
    name="AI Integration Advisor",
    model=OpenAIChat(id="gpt-4o"),
    description="Suggests AI implementation strategies based on industry insights and company operations.",
    show_tool_calls=True,
    markdown=True,
)

def suggest_ai_integration(company_data: str, ai_strategy: str):
    query = f"""
    Based on the AI adoption strategy:
    
    - **Company Context:** {company_data}
    - **AI Strategy Summary:** {ai_strategy}
    
    Provide a structured AI implementation plan:
    1. **Step-by-step AI Integration**: List phases of AI adoption, from pilot testing to full deployment.
    2. **Technology & Infrastructure**: Recommend necessary AI tools, cloud platforms, and software.
    3. **Workforce & Training**: Suggest ways to upskill employees for AI adoption.
    4. **Risk & Compliance Considerations**: Highlight data security, compliance, and ethical concerns.
    5. **KPIs for Success**: Define measurable AI performance indicators.
    
    The output should be detailed, actionable, and specific to the business domain.
    """
    return ai_integration_agent.print_response(query)


##############################
# 3️⃣ Revenue Growth Agent    #
##############################
revenue_growth_agent = Agent(
    name="Revenue Growth Agent",
    model=OpenAIChat(id="gpt-4o"),
    description="Identifies AI-driven opportunities to enhance revenue and efficiency.",
    show_tool_calls=True,
    markdown=True,
)

def identify_revenue_opportunities(company_data: str, ai_strategy: str):
    query = f"""
    You are an AI business analyst tasked with identifying AI-driven revenue growth opportunities for:
    
    - **Company Overview:** {company_data}
    - **AI Strategy:** {ai_strategy}
    
    Provide:
    1. **AI Monetization Strategies**: Explain how AI can create new revenue streams (e.g., AI-driven products, services, or data monetization).
    2. **Cost Reduction & Efficiency Gains**: Highlight AI automation that lowers operational costs.
    3. **Market Expansion**: Discuss how AI can help enter new markets or scale offerings.
    4. **Competitive Positioning**: Compare with industry leaders and suggest differentiation tactics.
    
    Ensure detailed, actionable insights with real-world examples where applicable.
    """
    return revenue_growth_agent.print_response(query)


##############################
# 4️⃣ Report Generation Agent #
##############################
def generate_report(company_name: str, ai_strategy: str, ai_integration: str, revenue_opportunities: str):
    report_content = f"""
    # AI Strategy Report for {company_name}
    
    ## AI Adoption Strategy
    {ai_strategy}
    
    ## AI Implementation Plan
    {ai_integration}
    
    ## Revenue Growth Opportunities
    {revenue_opportunities}
    
    """
    
    # Convert to Markdown
    markdown_report = markdown2.markdown(report_content)
    
    # Convert Markdown to PDF
    pdfkit.from_string(markdown_report, f"{company_name}_AI_Report.pdf")
    
    return f"Report generated: {company_name}_AI_Report.pdf"


###########################
# Example Usage           #
###########################
# if __name__ == "__main__":
#     company_name = "Tesla"
#     company_data = "Tesla specializes in electric vehicles and AI-powered self-driving technology."
#     industry_trends = "Latest AI advancements in autonomous driving and battery optimization."
#     ai_use_cases = "AI used in predictive maintenance, customer behavior analysis, and automation."
#     competitor_analysis = "Ford and GM are integrating AI into manufacturing and autonomous vehicle tech."
    
#     print("Generating AI Strategy...")
#     ai_strategy = generate_ai_strategy(company_data, industry_trends, ai_use_cases, competitor_analysis)
    
#     print("\nSuggesting AI Integration Plan...")
#     ai_integration = suggest_ai_integration(company_data, ai_strategy)
    
#     print("\nIdentifying Revenue Growth Opportunities...")
#     revenue_opportunities = identify_revenue_opportunities(company_data, ai_strategy)
    
#     print("\nGenerating Final Report...")
#     generate_report(company_name, ai_strategy, ai_integration, revenue_opportunities)