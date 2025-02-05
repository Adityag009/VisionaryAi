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
    query = f"Latest AI advancements and technology trends in {industry}."
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
    query = f"How is AI being used in {industry}? Provide real-world AI applications and case studies."
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
    query = f"How is {company_name} leveraging AI in its business operations? Find relevant reports and case studies."
    return competitive_analysis_agent.print_response(query)


###########################
# Example Usage           #
###########################
if __name__ == "__main__":
    industry = "Healthcare"
    print("Industry Trends:")
    get_industry_trends(industry)
    
    print("\nAI Use Cases:")
    get_ai_use_cases(industry)
    
    competitor = "Pfizer"
    print("\nCompetitor AI Strategies:")
    get_competitor_ai_strategies(competitor)
