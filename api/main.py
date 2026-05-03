from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
import uuid, os, base64, shutil, sys
from datetime import datetime
from dotenv import load_dotenv
import groq
from google import genai
import fitz

# ── Setup ─────────────────────────────────────────────────────────────────
load_dotenv()

app = FastAPI(title="Explainly API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.models import SessionLocal, Session as SessionModel, Response as ResponseModel

# ── AI Clients ────────────────────────────────────────────────────────────
groq_client   = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))
gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── Database dependency ───────────────────────────────────────────────────
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════

def transcribe_audio(audio_path: str) -> str:
    with open(audio_path, "rb") as f:
        transcription = groq_client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=f,
            response_format="text"
        )
    return transcription

def pdf_to_image(pdf_path: str, output_path: str) -> str:
    doc = fitz.open(pdf_path)
    page = doc[0]
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    pix.save(output_path)
    return output_path

# ── Extract question text from image using Gemini ─────────────────────────
def extract_question_from_image(image_path: str) -> str:
    try:
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[{
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": image_data
                        }
                    },
                    {
                        "text": """Extract the math question from this image exactly as written.
Return ONLY the question text — no explanation, no extra words.
If there are multiple questions extract all of them.
If you cannot find a clear question write: Question from uploaded image."""
                    }
                ]
            }]
        )
        return response.text.strip()

    except Exception as e:
        return f"Question from uploaded image (extraction failed: {str(e)})"
    
def parse_feedback(response_text: str) -> dict:
    feedback = {
        "score": 1,
        "representations_used": "None detected",
        "what_was_right": "",
        "what_to_improve": "",
        "ai_teacher_note": "",
        "language_detected": "English",
        "misconception_flag": False,
        "notation_errors": "None detected",
        "representation_strength": "ABSENT"
    }

    current_key = None
    current_value = []

    key_map = {
        "SCORE:": None,
        "REPRESENTATIONS_USED:": "representations_used",
        "WHAT_WAS_RIGHT:": "what_was_right",
        "WHAT_TO_IMPROVE:": "what_to_improve",
        "TEACHER_NOTE:": "ai_teacher_note",
        "LANGUAGE_DETECTED:": "language_detected",
        "MISCONCEPTION_FLAG:": None,
        "NOTATION_ERRORS:": "notation_errors",
        "REPRESENTATION_STRENGTH:": "representation_strength"
    }

    for line in response_text.strip().split('\n'):
        line = line.strip()
        matched = False

        for prefix, key in key_map.items():
            if line.startswith(prefix):
                if current_key and current_value:
                    feedback[current_key] = " ".join(current_value)
                    current_key = None
                    current_value = []

                value = line.replace(prefix, "").strip()

                if prefix == "SCORE:":
                    try:
                        feedback["score"] = int(value[0])
                    except:
                        feedback["score"] = 1
                elif prefix == "MISCONCEPTION_FLAG:":
                    feedback["misconception_flag"] = value.upper() == "YES"
                elif prefix in ["LANGUAGE_DETECTED:", "REPRESENTATION_STRENGTH:"]:
                    feedback[key] = value
                else:
                    current_key = key
                    current_value = [value] if value else []

                matched = True
                break

        if not matched and current_key and line:
            current_value.append(line)

    if current_key and current_value:
        feedback[current_key] = " ".join(current_value)

    return feedback

def analyse_with_gemini(
    question: str,
    transcript: str = None,
    image_path: str = None
) -> dict:

    # Build submission context based on what was provided
    if transcript and image_path:
        submission_context = f"""The student submitted BOTH a voice explanation AND written working.

THE STUDENT'S VERBAL EXPLANATION (transcribed from voice):
{transcript}

THE IMAGE shows the student's written working or canvas drawing.
Analyse BOTH the verbal explanation AND the written working together.
Cross reference them — do they match? Do they contradict each other?"""
        has_image = True

    elif transcript and not image_path:
        submission_context = f"""The student submitted a VOICE EXPLANATION ONLY.
No written working was provided.

THE STUDENT'S VERBAL EXPLANATION (transcribed from voice):
{transcript}

Evaluate based on the verbal explanation alone.
Note in your teacher note that no written working was provided."""
        has_image = False

    elif image_path and not transcript:
        submission_context = f"""The student submitted WRITTEN WORKING ONLY.
No voice explanation was provided.

THE IMAGE shows the student's written working or canvas drawing.
Evaluate based on the written working alone.
Note in your teacher note that no verbal explanation was provided."""
        has_image = True

    else:
        raise ValueError("At least one of transcript or image must be provided")

    prompt = f"""You are an expert mathematics educator with deep knowledge of
mathematics pedagogy, formative assessment, and teaching for understanding.
You believe mathematical proficiency means flexible thinking, conceptual
understanding, and the ability to represent mathematical ideas in multiple ways.

═══════════════════════════════════════════════════════════════
THE QUESTION THE TEACHER SET:
{question}

{submission_context}
═══════════════════════════════════════════════════════════════

PART 1 — IDENTIFY REPRESENTATIONS
Look carefully at what was submitted and identify which
mathematical representations the student used:

VISUAL MODELS (look in image if provided):
- Fraction strips or bar models
- Number lines
- Number bonds
- Area models or grid diagrams
- Part-whole diagrams
- Arrays or grouping diagrams
- Ratio tables or double number lines
- Geometric diagrams or constructions
- Tree diagrams or factor trees
- Place value charts
- Graphs or coordinate planes
- Any visual model the student invented

VERBAL REPRESENTATIONS (look in transcript if provided):
- Real life examples or analogies
- Describing a visual model they drew
- Using mathematical vocabulary correctly
- Explaining using because / since / this means
- Connecting to prior knowledge
- Self correcting during explanation

SYMBOLIC REPRESENTATIONS (look in image if provided):
- Standard algorithms or written calculations
- Equations or expressions
- Formal mathematical notation

═══════════════════════════════════════════════════════════════
PART 2 — EVALUATE ACROSS SIX DIMENSIONS
═══════════════════════════════════════════════════════════════

DIMENSION 1 — CONCEPTUAL UNDERSTANDING
Does the student understand the underlying concept?
- Do they explain WHAT the numbers or operations mean?
- Do they connect procedure to concept?
- Do they use because / since / therefore / this means?
- Correct answer with no conceptual explanation = WEAK understanding

DIMENSION 2 — PROCEDURAL ACCURACY
Are the mathematical steps correct?
- Is the final answer correct?
- Are intermediate steps mathematically valid?
- Are there calculation errors?
- CRITICAL: Did student reach correct answer through incorrect reasoning?

DIMENSION 3 — REPRESENTATIONAL FLEXIBILITY
Did the student use mathematical models?
- Did they use a visual model (fraction strip, number line, bar model)?
- Does the visual model correctly represent the concept?
- Did they connect multiple representations?
- Symbolic only with no visual or verbal reasoning = LIMITED flexibility

DIMENSION 4 — MATHEMATICAL REASONING AND COMMUNICATION
Can the student explain their thinking?
- Do they use because / since / therefore / this means?
- Do they explain WHY each step follows from the previous?
- Do they use correct mathematical vocabulary?

DIMENSION 5 — MISCONCEPTION DETECTION
Look carefully for hidden misconceptions:
- Does explanation reveal misunderstanding even if answer is correct?
- Are there incorrect steps that accidentally produced correct answer?
- Common misconceptions to watch for:
  * Fractions: larger denominator = larger fraction
  * Ratios: confusing ratio with fraction
  * Angles: thinking angles multiply rather than add to 180
  * Algebra: adding instead of multiplying both sides
  * Percentages: confusing percentage increase with percentage of
  * Decimals: thinking 0.10 is larger than 0.9
  * Fractions: adding numerators and denominators separately
- Cross reference verbal and visual — they may contradict

DIMENSION 6 — MATHEMATICAL NOTATION ACCURACY
Look carefully for notation errors:

EQUALS SIGN MISUSE (most common):
- Using = as then or next step or gives
  Example: 180 ÷ 10 = 18 × 2 = 36
  This implies 180 ÷ 10 = 36 which is FALSE
  Correct: write on separate lines

FRACTION NOTATION:
- Adding fractions as 1/2 + 1/3 = 2/5

ALGEBRAIC NOTATION:
- Writing 2x = 10 = x = 5

RATIO NOTATION:
- Confusing ratio a:b with fraction a/b

FOR EACH NOTATION ERROR:
1. Identify exactly what was written
2. Explain what it mathematically means as written
3. Explain what the student likely intended
4. Is this a notation habit or a conceptual gap?

A student with perfect steps but incorrect notation
should receive SCORE 3 maximum.

═══════════════════════════════════════════════════════════════
SCORING RUBRIC:
═══════════════════════════════════════════════════════════════

SCORE 4 — Flexible and Deep Understanding
✓ Correct answer with correct mathematical steps
✓ Clear conceptual explanation — explains WHY not just WHAT
✓ Uses at least one appropriate visual or verbal model
✓ Uses mathematical vocabulary accurately
✓ No misconceptions detected
✓ Mathematical notation is accurate throughout

SCORE 3 — Procedural Competence with Developing Understanding
✓ Correct or mostly correct answer
✓ Steps are mostly correct procedurally
✗ Explanation is procedural — explains WHAT not WHY
✗ Minor misconception or reasoning partially unclear
✗ Minor notation errors present

SCORE 2 — Partial Understanding with Significant Gaps
✗ Answer may be incorrect OR steps contain errors
✗ Reasoning confused or incomplete
✗ Clear misconception present
✗ Significant notation errors that change mathematical meaning

SCORE 1 — Limited Understanding Shown
✗ Incorrect answer with incorrect or absent working
✗ No reasoning present
✗ Fundamental misconception detected
✗ Cannot connect explanation to mathematical work

═══════════════════════════════════════════════════════════════
LANGUAGE AND TONE:
═══════════════════════════════════════════════════════════════

The student may respond in English, Roman Urdu, Sindhi, Arabic, or mixed.
Evaluate mathematical understanding regardless of language.
Always respond in the SAME language the student used.
You are writing for a child aged 10-16.
Be warm, encouraging, and specific.
Never use the word wrong — use needs refinement instead.
Always acknowledge any visual model the student used by name.

═══════════════════════════════════════════════════════════════
RESPOND IN THIS EXACT FORMAT — NO EXTRA TEXT:
═══════════════════════════════════════════════════════════════

SCORE: [1, 2, 3, or 4]

REPRESENTATIONS_USED: [List visual models and representations identified.
If none write: Symbolic only — no visual model detected]

WHAT_WAS_RIGHT: [One or two encouraging sentences. Reference:
- The visual model they used if any — name it specifically
- Their actual words or written steps
- What their reasoning showed about understanding]

WHAT_TO_IMPROVE: [One or two specific sentences. Priority:
1. Misconception — name it clearly and kindly
2. Notation error — explain what they wrote and what it means
3. Reasoning gap — explain WHY not just WHAT
4. Missing visual model — suggest a specific model
5. Procedural error — point to specific step
Never say wrong. Never give the direct answer.]

TEACHER_NOTE: [Two sentences for teacher only:
Sentence 1 — Is understanding genuine/conceptual or procedural?
Any misconception even with correct answer? Notation errors?
Sentence 2 — What specific follow up would deepen understanding?
Note if only voice or only written was submitted.]

LANGUAGE_DETECTED: [English / Roman Urdu / Sindhi / Arabic / Mixed]

MISCONCEPTION_FLAG: [YES / NO]

NOTATION_ERRORS: [List notation errors. Format each as:
Written: [what student wrote] | Means: [mathematical meaning] | Intended: [what student meant]
If none write: None detected — notation is mathematically accurate]

REPRESENTATION_STRENGTH: [STRONG / DEVELOPING / ABSENT
STRONG = used appropriate visual model with explanation
DEVELOPING = attempted visual model but incomplete or unexplained
ABSENT = no visual model used symbolic only]"""

    # Build content parts
    content_parts = []
    if has_image and image_path:
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        content_parts.append({
            "inline_data": {
                "mime_type": "image/png",
                "data": image_data
            }
        })
    content_parts.append({"text": prompt})

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[{"parts": content_parts}]
    )
    return parse_feedback(response.text)

# ── Scenario 1 — Analyse text only question using Groq Llama ──────────────
def analyse_question_text_only(question: str) -> str:
    try:
        prompt = f"""You are an expert curriculum designer and teacher coach
specialising in mathematics education and teaching for understanding.

A teacher has set this question for students:
"{question}"

Generate a teaching preparation brief to help the teacher
anticipate student thinking before sharing with the class.

Respond in this EXACT format:

MODEL_ANSWER: [A complete accurate answer a top student would give —
include conceptual explanation not just procedure]

EXPECTED_REPRESENTATIONS:
- [Visual model students might use 1]
- [Visual model students might use 2]
- [Symbolic approach students might use]

COMMON_MISCONCEPTIONS:
- [Misconception 1 — what students typically think and why]
- [Misconception 2]
- [Misconception 3]

COMMON_NOTATION_ERRORS:
- [Notation error 1 — what students write and what it actually means]
- [Notation error 2]

COMMON_REASONING_GAPS:
- [Where students say WHAT without explaining WHY]

SCORE_4_LOOKS_LIKE: [What a complete conceptual score 4 response includes]

SCORE_1_LOOKS_LIKE: [What a score 1 response typically says]"""

        response = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"Question analysis unavailable: {str(e)}"


# ── Scenario 2 — Analyse image question using Gemini Vision ──────────────
def analyse_question_with_image(question: str, image_path: str) -> str:
    try:
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        prompt = f"""You are an expert curriculum designer and teacher coach
specialising in mathematics education and teaching for understanding.

A teacher has set this question for students:
"{question}"

The image above shows the diagram, figure, or problem the students 
need to work with. Reference the specific values, angles, lines, 
and geometric relationships shown in the image throughout your analysis.

Generate a teaching preparation brief to help the teacher
anticipate student thinking before sharing with the class.

Respond in this EXACT format:

MODEL_ANSWER: [A complete accurate answer referencing the specific 
values and relationships shown in the image — include conceptual 
explanation not just procedure. Name the specific angles, values, 
or measurements visible in the diagram.]

EXPECTED_REPRESENTATIONS:
- [Visual model students might use 1 — reference the diagram]
- [Visual model students might use 2]
- [Symbolic approach students might use]

COMMON_MISCONCEPTIONS:
- [Misconception 1 specific to this diagram — what students think and why]
- [Misconception 2]
- [Misconception 3]

COMMON_NOTATION_ERRORS:
- [Notation error 1 specific to this type of problem]
- [Notation error 2]

COMMON_REASONING_GAPS:
- [Where students say WHAT without explaining WHY for this specific problem]

SCORE_4_LOOKS_LIKE: [What a complete score 4 response includes —
reference specific values from the diagram]

SCORE_1_LOOKS_LIKE: [What a score 1 response typically says]"""

        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[{
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": image_data
                        }
                    },
                    {"text": prompt}
                ]
            }]
        )
        return response.text

    except Exception as e:
        return f"Question analysis unavailable: {str(e)}"

def generate_class_summary(question: str, responses: list) -> dict:
    scores = [r.score for r in responses]
    avg_score = sum(scores) / len(scores) if scores else 0
    score_distribution = {1: 0, 2: 0, 3: 0, 4: 0}
    for s in scores:
        if s in score_distribution:
            score_distribution[s] += 1

    misconception_count = sum(1 for r in responses if r.misconception_flag)
    representation_counts = {"STRONG": 0, "DEVELOPING": 0, "ABSENT": 0}
    for r in responses:
        if r.representation_strength in representation_counts:
            representation_counts[r.representation_strength] += 1

    submission_modes = {"both": 0, "voice_only": 0, "written_only": 0}
    for r in responses:
        if r.submission_mode in submission_modes:
            submission_modes[r.submission_mode] += 1

    student_summaries = []
    for i, r in enumerate(responses):
        student_summaries.append(f"""
Student {i+1} ({r.student_name}):
- Score: {r.score}
- Submission mode: {r.submission_mode}
- Transcript: {r.transcript[:200] if r.transcript else 'Written only — no transcript'}
- Representations used: {r.representations_used or 'None'}
- Representation strength: {r.representation_strength or 'ABSENT'}
- Misconception detected: {r.misconception_flag}
- Notation errors: {r.notation_errors or 'None'}
- What was right: {r.what_was_right or 'N/A'}
- What to improve: {r.what_to_improve or 'N/A'}
""")

    all_summaries = "\n".join(student_summaries)

    prompt = f"""You are an expert mathematics educator and instructional coach
specialising in formative assessment and responsive teaching.

You have received responses from {len(responses)} students to this question:
QUESTION: {question}

CLASS STATISTICS:
- Total responses: {len(responses)}
- Average score: {avg_score:.1f} / 4
- Score distribution: {score_distribution}
- Students with misconceptions flagged: {misconception_count}
- Representation strength: {representation_counts}
- Submission modes: {submission_modes}

INDIVIDUAL STUDENT RESPONSES:
{all_summaries}

Analyse ALL responses and generate class level insights
to help the teacher decide their next teaching move.

Respond in this EXACT format:

COMMON_MISCONCEPTIONS: [List misconceptions in multiple students with count.
If none write: No common misconceptions detected]

COMMON_NOTATION_ERRORS: [List notation errors appearing repeatedly.
Note if habit or conceptual gap. If none write: No common notation errors]

REASONING_PATTERN: [MOSTLY_CONCEPTUAL / MOSTLY_PROCEDURAL / MIXED —
one sentence explanation]

REPRESENTATION_PATTERN: [One sentence about how students represented
their thinking — which models were used most]

SUBMISSION_PATTERN: [One sentence about submission modes —
note if many students avoided voice or written work]

STUDENTS_NEEDING_SUPPORT: [List student names scoring 1 or 2
with their specific gap. If none write: All students scoring 3 or above]

CLASS_READINESS: [READY_TO_MOVE_FORWARD / NEEDS_RETEACHING /
NEEDS_SMALL_GROUP — one sentence justification]

SUGGESTED_NEXT_MOVE: [Two to three specific sentences. Name the
specific teaching strategy, visual model, or classroom activity.
Reference actual misconceptions and patterns found.
Be specific about whole class vs small group vs individual action.]"""

    response = groq_client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.choices[0].message.content
    summary = {
        "total_responses": len(responses),
        "average_score": round(avg_score, 1),
        "score_distribution": score_distribution,
        "misconception_count": misconception_count,
        "representation_counts": representation_counts,
        "submission_modes": submission_modes,
        "common_misconceptions": "",
        "common_notation_errors": "",
        "reasoning_pattern": "",
        "representation_pattern": "",
        "submission_pattern": "",
        "students_needing_support": "",
        "class_readiness": "",
        "suggested_next_move": "",
        "raw_analysis": raw
    }

    for line in raw.strip().split('\n'):
        if line.startswith("COMMON_MISCONCEPTIONS:"):
            summary["common_misconceptions"] = line.replace("COMMON_MISCONCEPTIONS:", "").strip()
        elif line.startswith("COMMON_NOTATION_ERRORS:"):
            summary["common_notation_errors"] = line.replace("COMMON_NOTATION_ERRORS:", "").strip()
        elif line.startswith("REASONING_PATTERN:"):
            summary["reasoning_pattern"] = line.replace("REASONING_PATTERN:", "").strip()
        elif line.startswith("REPRESENTATION_PATTERN:"):
            summary["representation_pattern"] = line.replace("REPRESENTATION_PATTERN:", "").strip()
        elif line.startswith("SUBMISSION_PATTERN:"):
            summary["submission_pattern"] = line.replace("SUBMISSION_PATTERN:", "").strip()
        elif line.startswith("STUDENTS_NEEDING_SUPPORT:"):
            summary["students_needing_support"] = line.replace("STUDENTS_NEEDING_SUPPORT:", "").strip()
        elif line.startswith("CLASS_READINESS:"):
            summary["class_readiness"] = line.replace("CLASS_READINESS:", "").strip()
        elif line.startswith("SUGGESTED_NEXT_MOVE:"):
            summary["suggested_next_move"] = line.replace("SUGGESTED_NEXT_MOVE:", "").strip()

    return summary

# ══════════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    return {"status": "ok", "app": "Explainly", "version": "1.0"}

@app.get("/sessions")
def get_all_sessions(db: Session = Depends(get_db)):
    sessions = db.query(SessionModel).order_by(
        SessionModel.created_at.desc()
    ).all()
    return [
        {
            "session_id": s.id,
            "question": s.question,
            "question_image_filename": s.question_image_filename,
            "created_at": s.created_at,
            "student_link": s.student_link
        }
        for s in sessions
    ]

@app.post("/sessions")
async def create_session(
    question: str = Form(None),
    question_image: UploadFile = File(None),
    db: Session = Depends(get_db)
):
    # Validate — exactly one must be provided
    has_text = question and question.strip()
    has_image = question_image and question_image.filename

    if not has_text and not has_image:
        raise HTTPException(
            status_code=400,
            detail="Please provide either a question text or a question image"
        )

    if has_text and has_image:
        raise HTTPException(
            status_code=400,
            detail="Please provide either question text OR an image — not both"
        )

    question_image_filename = None
    image_path = None

    # ── SCENARIO 2 — Image only ───────────────────────────────────────────
    if has_image:
        ext = question_image.filename.split(".")[-1].lower()
        question_image_filename = f"question_{uuid.uuid4()}.{ext}"
        image_path = os.path.join(UPLOAD_DIR, question_image_filename)

        with open(image_path, "wb") as f:
            shutil.copyfileobj(question_image.file, f)

        # Convert PDF to image if needed
        if ext == "pdf":
            png_filename = question_image_filename.replace(".pdf", ".png")
            png_path = os.path.join(UPLOAD_DIR, png_filename)
            pdf_to_image(image_path, png_path)
            question_image_filename = png_filename
            image_path = png_path

        # Extract question text from image using Gemini
        question = extract_question_from_image(image_path)

        # Analyse using Gemini (has image context)
        question_analysis = analyse_question_with_image(question, image_path)

    # ── SCENARIO 1 — Text only ────────────────────────────────────────────
    else:
        # Analyse using Groq Llama (text only)
        question_analysis = analyse_question_text_only(question)

    # Generate unique student link
    student_link = str(uuid.uuid4())

    # Save to database
    session = SessionModel(
        question=question,
        student_link=student_link,
        question_image_filename=question_image_filename,
        model_answer=question_analysis,
        created_at=datetime.utcnow()
    )
    db.add(session)
    db.commit()
    db.refresh(session)

    return {
        "session_id": session.id,
        "student_link": student_link,
        "question": question,
        "question_analysis": question_analysis
    }
@app.get("/sessions/{session_id}")
def get_session(session_id: int, db: Session = Depends(get_db)):
    session = db.query(SessionModel).filter(
        SessionModel.id == session_id
    ).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": session.id,
        "question": session.question,
        "question_image_filename": session.question_image_filename,
        "student_link": session.student_link
    }

@app.post("/sessions/{session_id}/respond")
async def submit_response(
    session_id: int,
    student_name: str = Form(...),
    audio_file: UploadFile = File(None),
    canvas_image: UploadFile = File(None),
    uploaded_photo: UploadFile = File(None),
    db: Session = Depends(get_db)
):
    session = db.query(SessionModel).filter(
        SessionModel.id == session_id
    ).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    has_audio  = audio_file and audio_file.filename
    has_canvas = canvas_image and canvas_image.filename
    has_photo  = uploaded_photo and uploaded_photo.filename

    if not has_audio and not has_canvas and not has_photo:
        raise HTTPException(
            status_code=400,
            detail="At least one of audio, canvas, or photo must be provided"
        )

    # Save audio
    audio_filename = None
    transcript = None
    if has_audio:
        audio_ext = audio_file.filename.split(".")[-1].lower()
        audio_filename = f"audio_{uuid.uuid4()}.{audio_ext}"
        audio_path = os.path.join(UPLOAD_DIR, audio_filename)
        with open(audio_path, "wb") as f:
            shutil.copyfileobj(audio_file.file, f)
        transcript = transcribe_audio(audio_path)

    # Save canvas
    canvas_filename = None
    canvas_path = None
    if has_canvas:
        canvas_filename = f"canvas_{uuid.uuid4()}.png"
        canvas_path = os.path.join(UPLOAD_DIR, canvas_filename)
        with open(canvas_path, "wb") as f:
            shutil.copyfileobj(canvas_image.file, f)

    # Save uploaded photo
    uploaded_image_filename = None
    uploaded_path = None
    if has_photo:
        ext = uploaded_photo.filename.split(".")[-1].lower()
        uploaded_image_filename = f"photo_{uuid.uuid4()}.{ext}"
        uploaded_path = os.path.join(UPLOAD_DIR, uploaded_image_filename)
        with open(uploaded_path, "wb") as f:
            shutil.copyfileobj(uploaded_photo.file, f)
        if ext == "pdf":
            png_filename = uploaded_image_filename.replace(".pdf", ".png")
            png_path = os.path.join(UPLOAD_DIR, png_filename)
            pdf_to_image(uploaded_path, png_path)
            uploaded_image_filename = png_filename
            uploaded_path = png_path

    # Determine image to analyse — uploaded photo takes priority over canvas
    image_to_analyse = uploaded_path or canvas_path

    # Determine submission mode
    if transcript and image_to_analyse:
        submission_mode = "both"
    elif transcript:
        submission_mode = "voice_only"
    else:
        submission_mode = "written_only"

    # Analyse with Gemini
    feedback = analyse_with_gemini(
        question=session.question,
        transcript=transcript,
        image_path=image_to_analyse
    )

    # Save to database
    response = ResponseModel(
        session_id=session_id,
        student_name=student_name,
        submission_mode=submission_mode,
        audio_filename=audio_filename,
        canvas_image_filename=canvas_filename,
        uploaded_image_filename=uploaded_image_filename,
        transcript=transcript,
        score=feedback["score"],
        what_was_right=feedback["what_was_right"],
        what_to_improve=feedback["what_to_improve"],
        ai_teacher_note=feedback["ai_teacher_note"],
        language_detected=feedback["language_detected"],
        representations_used=feedback["representations_used"],
        representation_strength=feedback["representation_strength"],
        misconception_flag=feedback["misconception_flag"],
        notation_errors=feedback["notation_errors"],
        submitted_at=datetime.utcnow()
    )
    db.add(response)
    db.commit()
    db.refresh(response)

    return {
        "response_id": response.id,
        "transcript": transcript,
        "score": feedback["score"],
        "what_was_right": feedback["what_was_right"],
        "what_to_improve": feedback["what_to_improve"],
        "language_detected": feedback["language_detected"],
        "representations_used": feedback["representations_used"],
        "representation_strength": feedback["representation_strength"],
        "misconception_flag": feedback["misconception_flag"],
        "notation_errors": feedback["notation_errors"],
        "submission_mode": submission_mode
    }

@app.get("/sessions/{session_id}/responses")
def get_responses(session_id: int, db: Session = Depends(get_db)):
    responses = db.query(ResponseModel).filter(
        ResponseModel.session_id == session_id
    ).order_by(ResponseModel.submitted_at.asc()).all()

    return [
        {
            "response_id": r.id,
            "student_name": r.student_name,
            "submission_mode": r.submission_mode,
            "transcript": r.transcript,
            "score": r.score,
            "what_was_right": r.what_was_right,
            "what_to_improve": r.what_to_improve,
            "ai_teacher_note": r.ai_teacher_note,
            "language_detected": r.language_detected,
            "representations_used": r.representations_used,
            "representation_strength": r.representation_strength,
            "misconception_flag": r.misconception_flag,
            "notation_errors": r.notation_errors,
            "canvas_image_filename": r.canvas_image_filename,
            "uploaded_image_filename": r.uploaded_image_filename,
            "follow_up_flag": r.follow_up_flag,
            "teacher_private_note": r.teacher_private_note,
            "teacher_annotated": r.teacher_annotated,
            "submitted_at": r.submitted_at
        }
        for r in responses
    ]

@app.patch("/responses/{response_id}")
async def annotate_response(
    response_id: int,
    teacher_private_note: str = Form(None),
    follow_up_flag: bool = Form(None),
    db: Session = Depends(get_db)
):
    response = db.query(ResponseModel).filter(
        ResponseModel.id == response_id
    ).first()
    if not response:
        raise HTTPException(status_code=404, detail="Response not found")

    if teacher_private_note is not None:
        response.teacher_private_note = teacher_private_note
    if follow_up_flag is not None:
        response.follow_up_flag = follow_up_flag

    response.teacher_annotated = True
    db.commit()

    return {"message": "✅ Response annotated successfully"}

@app.get("/sessions/{session_id}/class-summary")
def get_class_summary(session_id: int, db: Session = Depends(get_db)):
    session = db.query(SessionModel).filter(
        SessionModel.id == session_id
    ).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    responses = db.query(ResponseModel).filter(
        ResponseModel.session_id == session_id
    ).all()

    if not responses:
        raise HTTPException(
            status_code=404,
            detail="No responses yet for this session"
        )

    summary = generate_class_summary(session.question, responses)

    return {
        "session_id": session_id,
        "question": session.question,
        "class_summary": summary
    }