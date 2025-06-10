



# 🎯 ResuMatch AI: Redefining Job-Candidate Synergy with Intelligent Analysis

**ResuMatch AI** is an intelligent resume screening system that leverages **Natural Language Processing** (NLP) and **zero-shot classification** (via RoBERTa) to automatically analyze and rank resumes based on their relevance to a given job description. This tool is designed for recruiters, HR professionals, and hiring platforms seeking efficiency, precision, and insight.

---

## 🚀 Aim

To develop an AI-powered resume shortlisting tool that reduces human effort, enhances accuracy in candidate screening, and visually communicates which resumes best match a job's requirements — **all without manual tagging or rule-based filters**.

---

## ✨ Special Features

* ✅ **Zero-Shot Learning (RoBERTa):** No model fine-tuning needed — evaluates resume relevance directly using state-of-the-art transformer models.
* 📄 **PDF Resume Parsing:** Efficient extraction of raw text using `PyPDF2`.
* 📌 **Skill and Experience Extraction:** Identifies important skills and calculates experience automatically using NLP and regex.
* 📊 **Scoring & Ranking Engine:**

  * Weights applied to skills match, experience, and semantic similarity.
  * Produces ranked results with interpretable breakdowns.
* 📈 **Visual Analytics Dashboard:** View top skills and experience ranges across top candidates via clean, chart-based summaries.
* 🌐 **JSON + HTML Support:** Works with both frontend dashboards and backend API clients.
* 📁 **Batch Resume Processing:** Accepts multiple PDF files in a single session.
* 🧠 **Stopword Filtering + Smart Tokenization** using `nltk`.

---

## 🎯 Uniqueness

* Uses **zero-shot classification** rather than traditional keyword matching — enabling **semantic-level understanding** of candidate-job relevance.
* No training dataset required — immediate, explainable results with any job description.
* Lightweight deployment with **Flask**, no need for heavy cloud infrastructure.
* Visual summaries help recruiters **quickly interpret the hiring landscape**.

---

## 💡 Advantages

* 🕒 **Time-Saving:** Automatically ranks resumes from large applicant pools.
* 🧠 **Intelligent Matching:** Captures hidden relevance, not just surface-level keywords.
* 🎨 **Insightful Charts:** Breakdown of top skills and experience groups helps spot trends.
* 🧰 **Customizable Scoring:** Weights for different criteria can be easily adjusted.
* 💻 **Developer-Friendly:** RESTful JSON endpoints make integration easy for web or mobile apps.

---

## 🔧 Installation

> Make sure you have **Python 3.8+** installed.

1. Clone the repository:

```bash
git clone https://github.com/your-username/resumatch-ai.git
cd resumatch-ai
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Download required NLTK stopwords:

```bash
python -c "import nltk; nltk.download('stopwords')"
```

---

## ▶️ Run the App

```bash
python app.py
```

Then visit [http://localhost:5000](http://localhost:5000) in your browser.

---

## 🧪 How It Works

1. Upload PDF resumes through the UI or API.
2. Paste a job description.
3. System extracts:

   * Skills
   * Experience durations
   * Job relevance (via RoBERTa zero-shot classification)
4. Calculates weighted scores and displays:

   * Ranked top 10 resumes
   * Pie/bar charts for skill & experience distribution

---

## 📤 API Endpoint: `/process`

### Request:

```json
{
  "pdf_directory": "uploads/resumes",
  "advertisement": "We are hiring a Data Analyst with 2+ years experience in SQL, Python, and data visualization.",
  "criteria_weights": {
    "skills": 0.5,
    "experience": 0.3,
    "match_score": 0.2
  }
}
```

### Response:

* `job_description`: Extracted job role
* `required_skills`: Skills list from JD
* `resumes`: Ranked results with metrics
* `charts`: Data for visual plots (skills, experience)

---

## 📂 Folder Structure

```
resumatch-ai/
├── app.py
├── requirements.txt
├── templates/
│   └── dashboard.html
├── uploads/
│   └── resumes/
├── static/
│   └── charts.js (optional for frontend)
└── README.md
```

---


