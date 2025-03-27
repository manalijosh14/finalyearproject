import os
import re
import logging
import csv
from tabulate import tabulate
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from collections import Counter
import nltk
from datetime import datetime
import torch
import warnings
import importlib.util
from PyPDF2 import PdfReader

# Check for required modules
required_modules = ['torch', 'transformers', 'PyPDF2', 'tabulate', 'nltk']
for module in required_modules:
    if importlib.util.find_spec(module) is None:
        logging.error(f"Module '{module}' is not installed. Please install it with 'pip install {module}'.")
        exit(1)

# Download stopwords from nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Initialize Zero-Shot-Classification pipeline
tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")
device = 0 if torch.cuda.is_available() else -1
zero_shot_classifier = pipeline(
    "zero-shot-classification",
    model=model,
    tokenizer=tokenizer,
    device=device
)
logging.info(f"Using device: {'GPU' if device == 0 else 'CPU'}")

# Load stopwords
stop_words = set(stopwords.words('english'))
custom_stopwords = set([
    "years", "year", "of", "in", "the", "to", "and", "a", "for", "with", "using", "by",
    "at", "on", "as", "an", "from", "for", "that", "i", "is", "are", "this", "it", "be", "or", "which"
])
all_stopwords = stop_words.union(custom_stopwords)

# Month-to-number mapping
MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
}

# Extract text from PDFs using PyPDF2
def pdf_to_text(pdf_file):
    text = ""
    try:
        with open(pdf_file, 'rb') as file:
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
    except Exception as e:
        logging.error(f"Error reading {pdf_file}: {e}")
    return text

# Extract resumes into an in-memory database (dictionary)
def extract_resumes(resume_directory):
    extracted_db = {}
    pdf_files = [f for f in os.listdir(resume_directory) if f.lower().endswith('.pdf')]
    for pdf_file in pdf_files:
        resume_path = os.path.join(resume_directory, pdf_file)
        text = pdf_to_text(resume_path)
        if text:
            extracted_db[resume_path] = text
        else:
            logging.warning(f"No text extracted from {pdf_file}")
    logging.info(f"Extracted {len(extracted_db)} resumes from {resume_directory}")
    return extracted_db, pdf_files

# Normalize and filter text
def filter_text(text):
    text = text.lower()
    words = re.findall(r'[a-z0-9]+(?:[-_+][a-z0-9]+)*', text)
    return [word for word in words if word not in all_stopwords]

# Calculate experience
def calculate_experience(start_date, end_date):
    try:
        start_date = start_date.strip().lower()
        end_date = end_date.strip().lower()
        start_parts = start_date.split()
        if len(start_parts) < 2:
            raise ValueError("Invalid start date format")
        start_month = MONTHS.get(start_parts[0], 1)
        start_year = int(start_parts[1])

        if end_date == "present":
            end_date = datetime.today()
            end_month, end_year = end_date.month, end_date.year
        else:
            end_parts = end_date.split()
            if len(end_parts) < 2:
                raise ValueError("Invalid end date format")
            end_month = MONTHS.get(end_parts[0], 1)
            end_year = int(end_parts[1])

        total_months = (end_year - start_year) * 12 + (end_month - start_month)
        if total_months < 0:
            raise ValueError("End date is earlier than start date")
        years = total_months // 12
        months = total_months % 12
        return years, months
    except Exception as e:
        logging.error(f"Error calculating experience: {e}")
        return 0, 0

# Extract phone number from text
def extract_phone_number(text):
    phone_pattern = r'(?:(?:\+\d{1,3}\s?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4})'
    match = re.search(phone_pattern, text)
    return match.group(0) if match else "Not Found"

# Extract skills, experience, and phone number from resume text
def extract_skills_experience(text, required_skills):
    filtered_words = filter_text(text)
    word_counts = Counter(filtered_words)
    required_skills_lower = [skill.lower() for skill in required_skills]
    key_skills = []

    for word in word_counts:
        if any(char.isdigit() for char in word):
            continue
        if word in required_skills_lower:
            key_skills.append(word)
        else:
            for skill in required_skills_lower:
                if word in skill or skill in word or word.replace('-', ' ') in skill or skill.replace(' ', '-') in word:
                    key_skills.append(skill)
                    break

    key_skills = list(set(key_skills))
    experience_matches = re.findall(r'(\w+\s\d{4})\s*[-to]\s*(\w+\s\d{4}|\s*present)', text, re.IGNORECASE)
    total_years, total_months = 0, 0
    for start_date, end_date in experience_matches:
        years, months = calculate_experience(start_date, end_date.strip())
        total_years += years
        total_months += months
    total_years += total_months // 12
    total_months = total_months % 12
    
    phone_number = extract_phone_number(text)
    return key_skills, total_years, total_months, phone_number

# Extract job details from manually entered advertisement
def extract_job_details(advertisement):
    advertisement = advertisement.lower()
    lines = advertisement.split('\n')

    # Extract job title
    job_title = "Unknown Job Title"
    for line in lines:
        job_title_match = re.search(r'(?:seeking|looking for|hiring|wanted|position|role|job)\s*(?:a|an)?\s*([a-z\s\-]+?)(?:\s*to|\s*with|\s*for|\s*in|\s*$)', line)
        if job_title_match:
            job_title = job_title_match.group(1).strip()
            break

    # Extract required years of experience
    required_years = 0.0
    for line in lines:
        exp_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:years?|yrs?)\s*(?:of)?\s*(?:experience)?', line)
        if exp_match:
            required_years = float(exp_match.group(1))
            break

    # Extract required skills
    required_skills = []
    skills_pattern = r'(?:required\s+skills|skills)\s*[:\-]?\s*([a-z0-9,\s()+\-]+)'
    for line in lines:
        skills_match = re.search(skills_pattern, line, re.IGNORECASE)
        if skills_match:
            skills_text = skills_match.group(1).strip()
            logging.info(f"Skills text matched: {skills_text}")
            skills_list = [skill.strip() for skill in skills_text.split(',')]
            for skill in skills_list:
                if not skill or re.match(r'^\d+$', skill) or 'experience' in skill.lower():
                    continue
                if '(' in skill and ')' in skill:
                    base_skill = skill.split('(')[0].strip()
                    options = re.findall(r'\b\w+\b', skill.split('(')[1].replace(')', '').replace('or', ','))
                    required_skills.append(base_skill)
                    required_skills.extend(options)
                else:
                    required_skills.append(skill)
            break

    # Fallback to common skills if no explicit skills found
    if not required_skills:
        common_skills = ['python', 'sql', 'java', 'javascript', 'aws', 'azure', 'gcp', 'spark', 'hadoop', 
                         'machine learning', 'data analysis', 'c++', 'kubernetes', 'docker', 'cybersecurity', 
                         'penetration testing', 'excel', 'tableau', 'linux', 'git']
        required_skills = [skill for skill in common_skills if skill in advertisement]

    required_skills = list(set(required_skills))
    return job_title, required_skills, required_years

# Zero-shot classification
def classify_resume_match(resume_text, job_description):
    candidate_labels = [job_description, "Not Relevant"]
    result = zero_shot_classifier(
        resume_text,
        candidate_labels=candidate_labels,
        hypothesis_template="This resume matches the role of {}.",
        truncation=True
    )
    best_label = result['labels'][0]
    match_score = result['scores'][0] if best_label == job_description else 1 - result['scores'][0]
    return best_label, match_score

# Process resumes and save top 10 to CSV
def process_resumes(resume_directory, job_description, required_skills, required_years):
    criteria_weights = {"skills": 0.4, "experience": 0.4, "match_score": 0.2}

    if not os.path.isdir(resume_directory):
        logging.error(f"Directory '{resume_directory}' does not exist.")
        return
    
    extracted_db, pdf_files = extract_resumes(resume_directory)
    if not extracted_db:
        logging.error("No resumes extracted from the directory.")
        return

    resume_results = []
    for pdf_file in pdf_files:
        resume_path = os.path.join(resume_directory, pdf_file)
        resume_text = extracted_db.get(resume_path)
        if not resume_text:
            logging.warning(f"No text available for {pdf_file}")
            continue
        
        best_label, match_score = classify_resume_match(resume_text, job_description)
        key_skills, total_years, total_months, phone_number = extract_skills_experience(resume_text, required_skills)

        skills_overlap = len(set(key_skills) & set([skill.lower() for skill in required_skills])) / max(len(required_skills), 1) if required_skills else 0
        exp_match = min(total_years / required_years, 1.0) if required_years > 0 else 1.0
        match_score = min(match_score, 1.0)

        weighted_score = (
            criteria_weights["skills"] * skills_overlap +
            criteria_weights["experience"] * exp_match +
            criteria_weights["match_score"] * match_score
        )

        resume_results.append({
            "pdf_name": pdf_file,
            "skills": ", ".join(key_skills) if key_skills else "None",
            "experience": f"{total_years} years, {total_months} months",
            "phone_number": phone_number,
            "match_score": match_score,
            "weighted_score": weighted_score
        })

    if not resume_results:
        logging.error("No resumes processed successfully.")
        return

    # Sort and take top 10
    resume_results = sorted(resume_results, key=lambda x: x["weighted_score"], reverse=True)[:10]
    headers = ["pdf_name", "skills", "experience", "phone_number", "match_score", "weighted_score"]
    top_10_results = [{k: r[k] for k in headers} for r in resume_results]
    
    # Display results
    print("Top 10 Resume Matches:")
    print(tabulate(top_10_results, headers="keys", tablefmt="grid"))

    # Save top 10 to CSV
    csv_file = "top_10_resumes.csv"
    try:
        with open(csv_file, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=headers)
            writer.writeheader()
            writer.writerows(top_10_results)
        logging.info(f"Top 10 results saved to {csv_file}")
    except Exception as e:
        logging.error(f"Failed to save top 10 results to CSV: {e}")

# Main execution
if __name__ == "__main__":
    print("Enter the job advertisement (up to 30 lines). Press Enter twice to finish:")
    advertisement_lines = []
    line_count = 0
    while line_count < 30:
        line = input()
        if line == "" and advertisement_lines:  # Stop if Enter is pressed twice
            break
        advertisement_lines.append(line)
        line_count += 1
    
    advertisement_text = "\n".join(advertisement_lines)
    job_description, required_skills, required_years = extract_job_details(advertisement_text)
    logging.info(f"Extracted Job Title: {job_description}")
    logging.info(f"Extracted Required Skills: {', '.join(required_skills) if required_skills else 'None'}")
    logging.info(f"Extracted Required Years: {required_years}")

    resume_directory = input("Enter path to candidate resume folder: ")
    process_resumes(resume_directory, job_description, required_skills, required_years)
