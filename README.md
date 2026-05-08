# Explainly
**Feedback that moves learning forward**

Atomcamp AI Bootcamp — Final Capstone Project

---

## What is Explainly?

Explainly is a full-stack AI web application for mathematics teachers
and students aged 10–16. Teachers post questions. Students respond using
voice, canvas drawing, or photo upload. Gemini 2.5 Flash evaluates every
response across six pedagogical dimensions and returns structured
feedback instantly — distinguishing misconceptions from notation errors
from careless slips.

---

## AI Architecture

Explainly uses a two-step agentic AI pipeline:

1. **Groq Whisper** (whisper-large-v3) transcribes student voice recordings
2. **Gemini 2.5 Flash** analyses the transcript and image together
   using a multimodal prompt across six dimensions
3. **Groq Llama 4 Scout** analyses text-only teacher questions

The system automatically selects the right model for each task and
falls back gracefully when a service is unavailable.

---

## Six Feedback Dimensions

| Dimension | What is evaluated |
|---|---|
| Conceptual Understanding | WHY not just WHAT |
| Procedural Accuracy | Are steps correct? |
| Representational Flexibility | Visual models used? |
| Mathematical Reasoning | Logical flow of explanation |
| Misconception Detection | Hidden misunderstandings |
| Notation Accuracy | Mathematical writing correct? |

---

## Error Classification

| Error Type | Definition | Teacher Response |
|---|---|---|
| MISCONCEPTION | Wrong conceptual understanding | Reteach the concept |
| NOTATION ERROR | Correct concept, wrong mathematical writing | Explicit notation instruction |
| CARELESS ERROR | Correct method, arithmetic slip | Ask student to check work |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI + Uvicorn (Python 3.13) |
| Database | SQLite (local) → Supabase PostgreSQL (production) |
| Speech to Text | Groq Whisper large-v3 |
| Vision AI | Google Gemini 2.5 Flash |
| Question AI | Groq Llama 4 Scout |
| Frontend | HTML5 + Vanilla JavaScript |
| Canvas | HTML5 Canvas API |
| Voice Recording | Browser MediaRecorder API |
| Deployment | Render + Supabase |

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/explainly.git
cd explainly
```

### 2. Create and activate virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install fastapi uvicorn python-multipart groq google-genai \
            sqlalchemy aiofiles python-dotenv pillow pymupdf \
            psycopg2-binary requests
```

### 4. Create .env file