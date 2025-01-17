import streamlit as st
import openai
import pdfplumber
import os
import pandas as pd
import urllib.parse
import re
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.linalg import norm
from dotenv import load_dotenv

st.cache_data.clear()  # Clear old cached data on every run

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Use a font that supports emojis (e.g., Arial Unicode MS)
plt.rcParams['font.family'] = 'Segoe UI Emoji'

# Set OpenAI API Key
#openai.api_key = ''

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

if openai.api_key is None:
    raise ValueError("⚠️ OPENAI_API_KEY is not set. Please set it before running the app.")
#print(openai.api_key)
# Function to get OpenAI embeddings
def get_embeddings(text, model="text-embedding-ada-002"):
    try:
        response = openai.Embedding.create(input=text, model=model)
        return response['data'][0]['embedding']
    except Exception as e:
        st.error(f"OpenAI Embedding Error: {str(e)}")
        return None

# Function to extract text from PDF
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text.strip()

# Function to create OpenAI prompt
def create_prompt(cv_text):
    return f"""
    Extract and structure the following CV into JSON format with these fields:
    - Personal Information: Name, Summary.
    - Technical Skills: Programming Languages, Tools, Competence Levels.
    - Education: Degree, Institution, Field of Study, Years Attended.
    - Professional Experience: Job Title, Organization, Duration, Responsibilities, Achievements.
    - Volunteer Experience: Role, Organization, Key Contributions.
    - Language Proficiency.
    - Industries Worked In.

    CV Text:
    {cv_text}
    """

# Function to parse the CV using OpenAI
def parse_cv_with_openai(cv_text):
    prompt = create_prompt(cv_text)
    completion_params = {
        'model': 'gpt-3.5-turbo',
        'messages': [
            {"role": "system", "content": "You are a CV parser. Only return a JSON object. Do NOT include Markdown, explanations, or any additional text."},
            {"role": "user", "content": prompt}
        ],
        'temperature': 0.0,
        'max_tokens': 1500
    }

    for attempt in range(3):  # Retry up to 3 times in case of errors
        try:
            print(openai.api_key)
            response = openai.ChatCompletion.create(**completion_params)
            
            # Extract and clean the response
            content = response['choices'][0]['message']['content'].strip()
            
            # Remove any Markdown artifacts (like ```json at the start and end)
            if content.startswith("```json"):
                content = content[7:]  # Remove ```json
            if content.endswith("```"):
                content = content[:-3]  # Remove ending ```

            # Ensure clean JSON response
            json_response = json.loads(content)
            return json_response

        except json.JSONDecodeError as e:
            st.error(f"OpenAI JSON Error: Failed to parse response. Retrying... (Attempt {attempt+1}/3)")
            time.sleep(2)  # Wait before retrying
        
        except Exception as e:
            st.error(f"OpenAI API Error: {str(e)}")
            return None
    
    st.error("OpenAI API Error: Unable to parse JSON after multiple attempts.")
    return None


# Function to compute cosine similarity
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

# Function to match candidates to job postings
def match_candidates(candidate_embedding, job_postings):
    job_postings['similarity_score'] = job_postings['embedding'].apply(
        lambda job_emb: cosine_similarity(candidate_embedding, job_emb)
    )
    return job_postings.sort_values(by='similarity_score', ascending=False).head(5)



# Function to process multiple CVs and match them with jobs
def process_cvs(uploaded_files, job_postings):
    all_matches = {}

    for uploaded_file in uploaded_files:
        with st.spinner(f"Processing {uploaded_file.name}..."):
            cv_text = extract_text_from_pdf(uploaded_file)
            candidate_data = parse_cv_with_openai(cv_text)

            if candidate_data:
                candidate_name = candidate_data.get("Personal Information", {}).get("Name", uploaded_file.name)
                candidate_embedding = get_embeddings(cv_text)

                # Get top job matches
                if candidate_embedding:
                    top_matches = match_candidates(candidate_embedding, job_postings)
                    all_matches[candidate_name] = top_matches
                else:
                    st.warning(f"Skipping {candidate_name} due to embedding error.")

    return all_matches

##------------------------------------AI-based feature------------------------------------  
# Function to generate AI job match summary
def generate_job_summary(job_title, job_description, candidate_name, candidate_cv):
    prompt = f"""
    You are an AI job expert. Summarize the key qualifications for the role of {job_title}.
    Then, suggest how {candidate_name} can improve their CV to match this role better.
    Also, recommend 2-3 similar jobs based on their profile.

    **Job Description:**
    {job_description}

    **Candidate CV Excerpt:**
    {candidate_cv}

    **Output Format:**
    - 🔹 **Job Summary:** [Brief summary]
    - 🏆 **How {candidate_name} Can Improve Their CV:** [Tips]
    - 🎯 **Alternative Job Recommendations:** [List 2-3 similar jobs]
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are a job-matching expert."},
                      {"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        st.error(f"AI Job Summary Error: {e}")
        return "AI-generated summary not available."


def generate_ai_summary(candidate_name, df):
    if df.empty:
        return "No job matches found."

    best_match = df.loc[df["Match Score (%)"].idxmax()]
    avg_score = df["Match Score (%)"].mean()
    job_titles = df["Job Title"].tolist()
    
    prompt = f"""
    You are an expert career advisor. Summarize the best job matches for {candidate_name} based on match scores and company fit.
    
    - Best Matched Job: {best_match['Job Title']} at {best_match['Company']} ({best_match['Match Score (%)']:.1f}% match).
    - Total Job Matches: {len(df)}.
    - Average Match Score: {avg_score:.1f}%.
    - Other Suitable Jobs: {', '.join(job_titles[:3])}.
    
    Provide a professional, concise summary.
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are an expert career advisor."},
                  {"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=150
    )
    
    return response["choices"][0]["message"]["content"].strip()

# Function to analyze candidate skills vs job requirements
def analyze_skills_with_ai(cv_text, job_description):
    prompt = f"""
    Analyze the candidate's CV and compare it with a job description.
    
    **Candidate CV:**
    {cv_text}
    
    **Job Description:**
    {job_description}
    
    - Identify key skills from the CV.
    - Identify required skills in the job description.
    - Highlight missing or less developed skills.
    
    Provide a concise analysis with bullet points.
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a career advisor specializing in skill analysis."},
                  {"role": "user", "content": prompt}],
        temperature=0.6,
        max_tokens=200
    )
    
    return response["choices"][0]["message"]["content"].strip()

# Function to suggest resume improvements
def resume_improvement_suggestions(cv_text):
    prompt = f"""
    You are an expert career coach. Provide concise improvement suggestions for this candidate's CV:
    
    **Candidate's CV:**
    {cv_text}
    
    - Identify weak or unclear phrasing.
    - Suggest more impactful language.
    - Provide 3 actionable improvements.
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a resume expert."},
                  {"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=200
    )
    
    return response["choices"][0]["message"]["content"].strip()


# Function to display candidate matches
def display_matches(candidate_name, top_matches, candidate_cv_text):
    st.subheader(f"Job Matches for {candidate_name}")

    # Create a copy to avoid SettingWithCopyWarning
    df = top_matches[['Roll', 'Company', 'Plats', 'Senioritetsnivå', 
                      'Distansarbete', 'Uppdragsperiod', 'Ansökningstiden löper ut', 
                      'similarity_score', 'Link']].copy()

    # Rename columns for readability
    df.rename(columns={
        'Roll': 'Job Title', 
        'Plats': 'Location', 
        'similarity_score': 'Match Score (%)', 
        'Link': 'Job Link'
    }, inplace=True)

    # Convert similarity score to percentage format
    df["Match Score (%)"] = df["Match Score (%)"] * 100

    # Extracting correct URL from Markdown-style links
    def extract_url(text):
        if isinstance(text, str):
            match = re.search(r'\[(.*?)\]\((.*?)\)', text)  # Extracts [text](URL)
            if match:
                return match.group(2)  # Extract the URL part only
            elif text.startswith("http"):  # If already a direct link
                return text
        return None  # Return None for missing links
    

    df["Job Link"] = df["Job Link"].apply(extract_url)  # Assign the column properly

    # Ensure Safe Encoding of URLs*
    df["Job Link"] = df["Job Link"].apply(lambda x: urllib.parse.unquote(x) if x else None)

    # Remove localhost issues by ensuring links start with http
    df["Job Link"] = df["Job Link"].apply(lambda x: x if x and x.startswith("http") else None)

    # *Convert links to Markdown clickable format
    df["Job Link"] = df.apply(
        lambda row: f'<a href="{row["Job Link"]}" target="_blank" style="color: blue; font-weight: bold;">🔗 View Job</a>'
        if row["Job Link"] else "No Link", axis=1
    )

    # Display table using `st.markdown()` for proper link rendering
    st.markdown(df.to_html(index=False, escape=False), unsafe_allow_html=True)

    # Apply the same strategy for "Apply for Jobs Below" section
    st.write("### 📌 Apply for Jobs Below:")
    for _, row in df.iterrows():
        job_title = row["Job Title"]
        company = row["Company"]
        job_link = row["Job Link"]

        if "href" in job_link:  # Check if job_link is a clickable hyperlink
            st.markdown(f"🔹 **Apply for {job_title} at {company}:** {job_link}", unsafe_allow_html=True)
        else:
            st.write(f"🔹 No job link available for **{job_title}**")
    
    # Filter UI
    st.write("### 📊 Job Match Score Ranking")

    df_sorted = df.sort_values(by="Match Score (%)", ascending=False).reset_index(drop=True)

    for _, row in df_sorted.iterrows():
        st.write(f"**{row['Job Title']} at {row['Company']}** ({row['Match Score (%)']:.1f}%)")
        st.progress(int(row["Match Score (%)"]))  # Progress bar based on match score


    # Filter UI
    st.write("### 🔍 Filter Jobs")
    job_title_filter = st.multiselect("Filter by Job Title", df["Job Title"].unique(), default=[], help="Select one or more job titles")
    company_filter = st.multiselect("Filter by Company", df["Company"].unique(), default=[], help="Select one or more companies")

    # Apply Filters (Only Show Jobs After Selection)
    if not job_title_filter and not company_filter:
        st.warning("⚠️ Select a **Job Title** or **Company** to view job matches.")
    else:
        # Apply Filters to the DataFrame
        filtered_df = df.copy()
        if job_title_filter:
            filtered_df = filtered_df[filtered_df["Job Title"].isin(job_title_filter)]
        if company_filter:
            filtered_df = filtered_df[filtered_df["Company"].isin(company_filter)]

        # Display Filtered Results Only
        if filtered_df.empty:
            st.warning("⚠️ No jobs match your filters. Try selecting different options.")
        else:
            st.markdown(filtered_df.to_html(index=False, escape=False), unsafe_allow_html=True)

            # ✅ **Show Job Descriptions Only When Filters Are Applied**
            st.write("### 📄 Job Descriptions")
            for _, row in filtered_df.iterrows():
                st.write(f"### {row['Job Title']} at {row['Company']}")
                st.write(f"📍 **Location:** {row['Location']}")
                st.write(f"💼 **Seniority Level:** {row['Senioritetsnivå']}")
                st.write(f"📊 **Match Score:** {row['Match Score (%)']:.2f}%")
                st.write(f"📝 **Job Description:** {row['Uppdragsperiod']}")
                st.write(f"📅 **Application Deadline:** {row['Ansökningstiden löper ut']}")
                st.markdown(row["Job Link"], unsafe_allow_html=True)

                # *AI-Powered Job Insights
                st.write("### 🔹 AI-Powered Job Insights")
                ai_job_summary = generate_job_summary(row['Job Title'], row['Uppdragsperiod'], candidate_name, candidate_cv_text)
                st.markdown(ai_job_summary)

                # AI-Powered Skill Gap Analysis
                st.write("### 📌 Skill Gap Analysis")
                ai_skill_analysis = analyze_skills_with_ai(candidate_cv_text, row['Uppdragsperiod'])
                st.markdown(ai_skill_analysis)

                st.markdown("---")  # Separate job descriptions for readability

    
    # 📌 **Job Matching Summary**
    st.write("### 📊 Job Matching Summary")

    total_matches = len(df)
    if total_matches > 0:
        best_match = df.loc[df["Match Score (%)"].idxmax()]
        avg_score = df["Match Score (%)"].mean()

        st.write(f"🔹 **Total Job Matches:** {total_matches}")
        st.write(f"🏆 **Best Match:** {best_match['Job Title']} at {best_match['Company']} ({best_match['Match Score (%)']:.1f}%)")
        st.write(f"📊 **Average Match Score:** {avg_score:.1f}%")
    else:
        st.write("⚠️ No job matches found.")

    


# Streamlit UI Setup
st.title("🔍 AI-Powered CV Matcher")
st.markdown("Upload CVs and find the best job matches based on skills, experience, and location.")

uploaded_files = st.file_uploader("Upload CV PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    job_postings = pd.read_excel("job_postings.xlsx")
    if 'embedding' not in job_postings.columns:
        job_postings['embedding'] = job_postings.apply(lambda row: get_embeddings(f"{row['Roll']} {row['Company']}"), axis=1)
    
    all_matches = {}
    for uploaded_file in uploaded_files:
        cv_text = extract_text_from_pdf(uploaded_file)
        candidate_embedding = get_embeddings(cv_text)
        if candidate_embedding:
            all_matches[uploaded_file.name] = match_candidates(candidate_embedding, job_postings)

    # Candidate selection
    selected_candidates = st.multiselect("Compare Candidates", list(all_matches.keys()))
    
    if selected_candidates:
        combined_jobs = pd.concat([all_matches[candidate] for candidate in selected_candidates], axis=0)
        all_job_titles = combined_jobs["Roll"].unique()
        heatmap_df = pd.DataFrame(index=all_job_titles)
        
        for candidate in selected_candidates:
            candidate_scores = all_matches[candidate][["Roll", "similarity_score"]].copy()
            candidate_scores.rename(columns={"similarity_score": f"{candidate}_Match Score"}, inplace=True)
            heatmap_df = heatmap_df.merge(candidate_scores.set_index("Roll"), left_index=True, right_index=True, how="left")
        
        heatmap_df.fillna(0, inplace=True)
        heatmap_df = heatmap_df.loc[heatmap_df.mean(axis=1).sort_values(ascending=False).index]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(heatmap_df, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
        ax.set_xlabel("Candidates")
        ax.set_ylabel("Job Title")
        ax.set_title("🔹 Skill Matching: Candidate vs. Job Match Scores")
        st.pyplot(fig)
    
    for candidate, top_matches in all_matches.items():
        with st.expander(f"📄 View Matches for {candidate}"):
            display_matches(candidate, top_matches, cv_text)
