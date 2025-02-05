import os
from phi.agent import Agent
from phi.model.openai import OpenAIChat
from typing import List
from pydantic import BaseModel, Field
import markdown2
import pdfkit

# Load environment variables (API keys, etc.)
from dotenv import load_dotenv
load_dotenv()



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
if __name__ == "__main__":
    company_name = "Tesla"
    company_data = "Tesla specializes in electric vehicles and AI-powered self-driving technology."
    industry_trends = "Latest AI advancements in autonomous driving and battery optimization."
    ai_use_cases = "AI used in predictive maintenance, customer behavior analysis, and automation."
    competitor_analysis = "Ford and GM are integrating AI into manufacturing and autonomous vehicle tech."
    
    print("Generating AI Strategy...")
    ai_strategy = generate_ai_strategy(company_data, industry_trends, ai_use_cases, competitor_analysis)
    
    print("\nSuggesting AI Integration Plan...")
    ai_integration = suggest_ai_integration(company_data, ai_strategy)
    
    print("\nIdentifying Revenue Growth Opportunities...")
    revenue_opportunities = identify_revenue_opportunities(company_data, ai_strategy)
    
    print("\nGenerating Final Report...")
    generate_report(company_name, ai_strategy, ai_integration, revenue_opportunities)