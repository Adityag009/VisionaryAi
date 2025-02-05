import streamlit as st
import pandas as pd
import json
import os
from VisionaryAgent import search_company, scrape_website, process_company_description, process_uploaded_document
from VisionaryAgent import get_industry_trends, get_ai_use_cases, get_competitor_ai_strategies
from VisionaryAgent import generate_ai_strategy, suggest_ai_integration, identify_revenue_opportunities, generate_report

# Define data storage paths
CSV_FILE = "user_data.csv"
JSON_FILE = "user_data.json"

# Function to save data to CSV
def save_data_csv(data):
    df = pd.DataFrame([data])
    if os.path.exists(CSV_FILE):
        df.to_csv(CSV_FILE, mode='a', header=False, index=False)
    else:
        df.to_csv(CSV_FILE, index=False)

# Function to save data to JSON
def save_data_json(data):
    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, "r") as file:
            existing_data = json.load(file)
    else:
        existing_data = []
    
    existing_data.append(data)
    with open(JSON_FILE, "w") as file:
        json.dump(existing_data, file, indent=4)

# Streamlit UI
def main():
    st.title("Visionary AI  by Giant Analytics")
    st.write("Fill in the details to generate an AI-driven business strategy report.")
    st.write("It uses SOTA (State-of-the-Art) Reasoning Models to provide cutting-edge insights and AI integration strategies.")
    # Collect User Information
    name = st.text_input("Name")
    email = st.text_input("Email")
    mobile = st.text_input("Mobile Number")
    company_name = st.text_input("Company Name")
    
    # Select method to provide company details
    input_method = st.radio("How would you like to provide company details?", 
                            ("Search by Name", "Website URL", "Manual Description", "Upload Document"))
    
    company_data = ""
    if input_method == "Search by Name":
        if st.button("Find Company Details"):
            company_data = search_company(company_name)
            st.write(company_data)
    elif input_method == "Website URL":
        website_url = st.text_input("Enter Website URL")
        if st.button("Scrape Website"):
            company_data = scrape_website(website_url)
            st.write(company_data)
    elif input_method == "Manual Description":
        company_data = st.text_area("Enter Company Description")
        if st.button("Process Description"):
            company_data = process_company_description(company_data)
            st.write(company_data)
    elif input_method == "Upload Document":
        uploaded_file = st.file_uploader("Upload PDF or PPT", type=["pdf", "pptx"])
        if uploaded_file is not None:
            company_data = process_uploaded_document(uploaded_file)
            st.write(company_data)
    
    if company_data:
        industry = st.text_input("Industry Type (e.g., Healthcare, Finance)")
        if st.button("Analyze Industry Trends"):
            industry_trends = get_industry_trends(industry)
            st.write(industry_trends)
        
        if st.button("Find AI Use Cases"):
            ai_use_cases = get_ai_use_cases(industry)
            st.write(ai_use_cases)
        
        competitor = st.text_input("Enter Competitor Name")
        if st.button("Analyze Competitor AI Strategies"):
            competitor_analysis = get_competitor_ai_strategies(competitor)
            st.write(competitor_analysis)
        
        if st.button("Generate AI Strategy"):
            ai_strategy = generate_ai_strategy(company_data, industry_trends, ai_use_cases, competitor_analysis)
            st.write(ai_strategy)
        
        if st.button("Suggest AI Integration Plan"):
            ai_integration = suggest_ai_integration(company_data, ai_strategy)
            st.write(ai_integration)
        
        if st.button("Identify Revenue Growth Opportunities"):
            revenue_opportunities = identify_revenue_opportunities(company_data, ai_strategy)
            st.write(revenue_opportunities)
        
        if st.button("Generate Final Report"):
            report_filename = generate_report(company_name, ai_strategy, ai_integration, revenue_opportunities)
            st.success(f"Report Generated: {report_filename}")
            
        # Save data to backend
        user_data = {
            "name": name,
            "email": email,
            "mobile": mobile,
            "company_name": company_name,
            "company_data": company_data,
            "industry": industry,
            "competitor": competitor,
            "ai_strategy": ai_strategy,
            "ai_integration": ai_integration,
            "revenue_opportunities": revenue_opportunities
        }
        save_data_csv(user_data)
        save_data_json(user_data)

if __name__ == "__main__":
    main()