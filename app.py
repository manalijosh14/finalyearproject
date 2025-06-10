from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import re
import PyPDF2
import logging
from datetime import datetime
import torch
import warnings
import nltk
from collections import Counter
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Download nltk data
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Initialize NLP components
try:
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
except Exception as e:
    logging.error(f"Failed to initialize model: {e}")
    zero_shot_classifier = None

# Month mapping
MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
}

# Helper Functions

def pdf_to_text(pdf_file):
    """Extract text from PDF file"""
    text = ""
    try:
        with open(pdf_file, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ''
    except Exception as e:
        logging.error(f"Error reading {pdf_file}: {e}")
    return text

def filter_text(text, all_stopwords):
    """Normalize and filter text"""
    text = text.lower()
    words = re.findall(r'\b\w+\b', text)
    return [word for word in words if word not in all_stopwords]

def calculate_experience(start_date, end_date):
    """Calculate years and months of experience"""
    try:
        start_date = start_date.strip().lower()
        end_date = end_date.strip().lower()
        start_parts = start_date.split()
        
        if len(start_parts) < 2:
            raise ValueError("Invalid start date format")
            
        start_month = MONTHS.get(start_parts[0], 1)
        start_year = int(start_parts[1])

        if end_date == "present":
            end_date = datetime.now()
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
        return total_months // 12, total_months % 12
    except Exception as e:
        logging.error(f"Error calculating experience: {e}")
        return 0, 0

def extract_skills_experience(text, required_skills, all_stopwords):
    """Extract skills and experience from text"""
    filtered_words = filter_text(text, all_stopwords)
    word_counts = Counter(filtered_words)
    key_skills = [word for word in word_counts if word in [skill.lower() for skill in required_skills]]
    
    experience_matches = re.findall(r'(\w+\s\d{4})\s*[-to]\s*(\w+\s\d{4}|\s*present)', text, re.IGNORECASE)
    total_years, total_months = 0, 0
    
    for start_date, end_date in experience_matches:
        years, months = calculate_experience(start_date, end_date.strip())
        total_years += years
        total_months += months
    
    total_years += total_months // 12
    total_months = total_months % 12
    
    return key_skills, total_years, total_months

def extract_job_details(advertisement):
    """Extract job details from advertisement text"""
    advertisement = advertisement.lower()
    
    # Extract job title
    job_title_match = re.search(r'(?:seeking|looking for|hiring)\s*(?:a|an)?\s*([a-z\s]+?)(?:\s*to|\s*with|\s*$)', advertisement)
    job_title = job_title_match.group(1).strip() if job_title_match else "Unknown Position"
    
    # Extract required experience
    exp_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:years?|yrs?)\s*(?:of)?\s*(?:experience)?', advertisement)
    required_years = float(exp_match.group(1)) if exp_match else 0.0
    
    # Extract required skills
    skills_keywords = r'(?:skills|required|must have|experience with|knowledge of)\s*[:\-]?\s*([a-z,\s()]+)'
    skills_match = re.search(skills_keywords, advertisement)
    required_skills = []
    
    if skills_match:
        skills_text = skills_match.group(1)
        skills_list = [skill.strip() for skill in skills_text.split(',')]
        
        for skill in skills_list:
            if '(' in skill and ')' in skill:
                base_skill = skill.split('(')[0].strip()
                options = re.findall(r'\b\w+\b', skill.split('(')[1].replace(')', '').replace('or', ','))
                required_skills.append(base_skill)
                required_skills.extend(options)
            else:
                required_skills.append(skill)
                
        required_skills = [s.strip() for s in required_skills if s.strip()]
    else:
        common_skills = ['python', 'sql', 'java', 'javascript', 'aws', 'spark', 'hadoop', 'machine learning', 'data analysis']
        required_skills = [skill for skill in common_skills if skill in advertisement]
    
    return job_title, required_skills, required_years

# Flask Routes

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/process', methods=['POST'])
def process_resumes_api():
    if not zero_shot_classifier:
        return jsonify({"success": False, "error": "NLP model not initialized"})
    
    try:
        data = request.json
        pdf_directory = data.get('pdf_directory', '')
        advertisement = data.get('advertisement', '')
        criteria_weights = data.get('criteria_weights', {
            "skills": 0.5,
            "experience": 0.3,
            "match_score": 0.2
        })
        
        # Load stopwords
        stop_words = set(stopwords.words('english'))
        custom_stopwords = set([
            "years", "year", "of", "in", "the", "to", "and", "a", "for", "with", "using", "by",
            "at", "on", "as", "an", "from", "for", "that", "i", "is", "are", "this", "it", "be", "or", "which",
            "education", "linkedin", "hobbies", "phone", "number", "achievement"
        ])
        all_stopwords = stop_words.union(custom_stopwords)
        
        # Extract job details
        job_description, required_skills, required_years = extract_job_details(advertisement)
        
        # Process resumes
        resume_results = []
        pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith('.pdf')]
        
        for pdf_file in pdf_files:
            resume_path = os.path.join(pdf_directory, pdf_file)
            logging.info(f"Processing resume: {pdf_file}")
            
            resume_text = pdf_to_text(resume_path)
            if not resume_text:
                continue
                
            # Classify resume match
            candidate_labels = [job_description, "Not Relevant"]
            result = zero_shot_classifier(
                resume_text,
                candidate_labels=candidate_labels,
                hypothesis_template="This resume matches the role of {}.",
                truncation=True
            )
            
            best_label = result['labels'][0]
            match_score = result['scores'][0] if best_label == job_description else 1 - result['scores'][0]
            
            # Extract skills and experience
            key_skills, total_years, total_months = extract_skills_experience(
                resume_text, 
                required_skills,
                all_stopwords
            )
            
            # Calculate scores
            skills_overlap = len(set(key_skills) & set([skill.lower() for skill in required_skills])) / max(len(required_skills), 1)
            exp_match = min(total_years / required_years, 1.5) if required_years > 0 else 1.0 if total_years > 0 else 0.0

            
            # Weighted score
            weighted_score = (
                criteria_weights.get("skills", 0) * skills_overlap +
                criteria_weights.get("experience", 0) * exp_match +
                criteria_weights.get("match_score", 0) * match_score
            ) * 100  # Convert to percentage
            
            resume_results.append({
                "pdf_name": pdf_file,
                "skills": key_skills,
                "skills_overlap": round(skills_overlap * 100, 1),
                "experience": f"{total_years} years, {total_months} months",
                "exp_match": round(exp_match * 100, 1),
                "match_score": round(match_score * 100, 1),
                "weighted_score": round(weighted_score, 1),
                "resume_text": resume_text[:1000] + "..." if len(resume_text) > 1000 else resume_text
            })
        
        # Sort and get top 10
        top_resumes = sorted(resume_results, key=lambda x: x["weighted_score"], reverse=True)[:10]
        
        # Generate charts data
        charts_data = generate_charts_data(top_resumes, required_skills)
        
        return jsonify({
            "success": True,
            "job_description": job_description,
            "required_skills": required_skills,
            "required_years": required_years,
            "criteria_weights": criteria_weights,
            "resumes": top_resumes,
            "charts": charts_data
        })
    
    except Exception as e:
        logging.error(f"Error processing resumes: {e}")
        return jsonify({"success": False, "error": str(e)})

def generate_charts_data(resumes, required_skills):
    """Generate data for charts visualization"""
    # Skills distribution
    skill_counts = Counter()
    for resume in resumes:
        for skill in resume['skills']:
            if skill in required_skills:
                skill_counts[skill] += 1
    
    # Experience distribution
    exp_bins = {"0-2": 0, "3-5": 0, "6-8": 0, "9+": 0}
    for resume in resumes:
        exp_text = resume['experience']
        years_match = re.search(r'(\d+) years', exp_text)
        if years_match:
            years = int(years_match.group(1))
            if years <= 2:
                exp_bins["0-2"] += 1
            elif 3 <= years <= 5:
                exp_bins["3-5"] += 1
            elif 6 <= years <= 8:
                exp_bins["6-8"] += 1
            else:
                exp_bins["9+"] += 1
    
    return {
        "skills": {
            "labels": list(skill_counts.keys()),
            "data": list(skill_counts.values())
        },
        "experience": {
            "labels": list(exp_bins.keys()),
            "data": list(exp_bins.values())
        }
    }

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        return jsonify({"success": False, "error": "No files uploaded"})
    
    files = request.files.getlist('files')
    upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'resumes')
    os.makedirs(upload_dir, exist_ok=True)
    
    saved_files = []
    for file in files:
        if file.filename == '':
            continue
        if file and file.filename.lower().endswith('.pdf'):
            filename = secure_filename(file.filename)
            file.save(os.path.join(upload_dir, filename))
            saved_files.append(filename)
    
    return jsonify({
        "success": True,
        "message": f"Successfully uploaded {len(saved_files)} files",
        "pdf_directory": upload_dir,
        "files": saved_files
    })

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(
        os.path.join(app.config['UPLOAD_FOLDER'], 'resumes'),
        filename,
        as_attachment=True
    )

def secure_filename(filename):
    """Sanitize filename to prevent directory traversal"""
    filename = os.path.basename(filename)
    filename = re.sub(r'[^\w\-_. ]', '', filename)
    return filename

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
