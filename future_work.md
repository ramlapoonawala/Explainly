# Future Work — Explainly

## Phase 1 — AI Model Optimisation (Next Priority)

### Model Comparison Testing
During development Gemini 2.5 Flash and Groq Llama 4 Scout
were evaluated. Gemini was selected for its superior
multimodal capability to analyse voice and image together.
The following models are candidates for Phase 2 evaluation:

| Model | Company | Why Test It |
|---|---|---|
| Qwen2-Math 72B | Alibaba | Specifically fine tuned for mathematics |
| DeepSeek R1 | DeepSeek | Strong reasoning chain — shows thinking step by step |
| Phi-3.5-mini | Microsoft | Small and fast — lower cost for scaling |
| Aya-23 | Cohere | Fine tuned for 23 languages including Urdu and Arabic |
| GPT-4o | OpenAI | Strong multimodal — benchmark comparison |

### Test Battery
A structured test battery of 8 cases has been designed
covering the specific failure modes identified during pilot:

- Test 1 — Decomposition strategy (creative method)
- Test 2 — Correct scientific notation chaining
- Test 3 — Clear fraction addition misconception
- Test 4 — Notation error without misconception
- Test 5 — Roman Urdu student response
- Test 6 — Careless error not misconception
- Test 7 — Difference of two squares correctly solved
- Test 8 — Written only submission with ambiguous working

Each model will be evaluated across five criteria:
- Correct score assigned (30%)
- Misconception flag correct (25%)
- Error type correct (20%)
- Working interpretation clear (15%)
- Feedback language appropriate (10%)

### Testing Approach
Models will be tested in Jupyter Notebook outside the
main project to avoid disrupting the working system.
The winning model will replace Gemini 2.5 Flash via
a one line change in main.py. OpenRouter API will be
used to access multiple models through a single endpoint.

### Hybrid Model Architecture
Based on pilot testing results a hybrid approach may
outperform a single model:
- Qwen2-Math for mathematical assessment and scoring
- Gemini for multilingual feedback generation
- DeepSeek R1 for complex reasoning edge cases

---

## Phase 2 — Fine Tuning Pipeline

### Data Collection
Teacher corrections collected through the UI will be
stored in a structured corrections table containing:
- Student input — transcript and canvas image
- AI output — score, flags, feedback
- Teacher correction — correct score and interpretation
- Correction type — strategy / misconception / notation / careless

### Fine Tuning Target
Once 500+ high quality corrections are collected:
- Fine tune a small open source model on Explainly data
- Compare fine tuned model against base Gemini
- Deploy if accuracy improvement exceeds 15%

### Quality Guidelines for Corrections
- Specific description of what AI misunderstood
- Consistent labelling across same error types
- Correct score assigned using rubric not gut feeling
- Only submit corrections when certain
- Balanced dataset across all correction types

---

## Phase 3 — AI Improvements

### Two Pass Analysis
Separate understanding from evaluation:
- Pass 1 — describe what method the student used
- Pass 2 — evaluate based on Pass 1 description
Prevents misreading creative strategies as errors

### Separate Voice and Written Scores
Voice score:   3/4 — good verbal explanation
Written score: 2/4 — working hard to follow
Combined:      3/4 — weighted average
Tells teacher exactly where the gap is

### Working Interpretation Field
New teacher-only field showing exactly how the AI
read the student's written working before evaluating.
Helps teacher spot AI misreadings immediately.

### Stronger Model for Written Only
Automatically switch to more powerful model when
student submits written only — hardest case for AI.

### Student History for Progress Tracking
Pass previous session scores into the prompt so AI
can acknowledge improvement and flag persistent
misconceptions across multiple sessions.

### Interactive Feedback Loop
Student responds to AI guiding question:
AI:      "What do you get when you multiply +2 by -2?"
Student: "-4"
AI:      "Exactly — so what should the middle term be?"
### Student Accounts
Persistent identity across sessions so history
is tracked automatically without teacher input.

### Voice Plus Written Nudge
When student selects written only mode show:
"Adding voice will get you more detailed feedback"

### Typed Working Input
Text area below canvas for students who prefer
typing mathematical working rather than drawing.

---

## Phase 5 — Teacher Analytics

### Filter by Follow Up Flag
Show only flagged students in responses view
so teacher can action them quickly after class.

### Export Class Results
Download session results as CSV or PDF for
school records and parent communication.

### Trend View
Average score per session plotted over the term
showing class progress against curriculum objectives.

### Misconception Frequency Map
Which misconceptions appear most across all sessions
mapped against the curriculum to identify teaching gaps.

### Auto Flag
Automatically pre-flag students scoring 1 with
misconception detected — teacher confirms or removes.

---

## Phase 6 — Session Management

### Delete Closed Sessions
Allow teachers to delete closed sessions after a
configurable retention period with a confirmation
warning showing how many student responses will
be permanently removed.

### Archive Before Delete
Export full session data as PDF before deletion
so teacher retains a record outside the system.

### Bulk Operations
Close all open sessions at end of term in one click.
Archive all closed sessions older than one year.

---

## Phase 7 — Platform Features

### Google Classroom Integration
Sync sessions with Google Classroom assignments.
Students access via Classroom — no separate link needed.

### LMS Integration
Moodle and Canvas integration for schools using
established learning management systems.

### Mobile App
Native iOS and Android app for smoother voice
recording and canvas drawing on mobile devices.

### Offline Mode
Core submission functionality works without internet.
Syncs to server when connection is restored.
Critical for low connectivity classrooms in Pakistan.

### Parent View
Read only access for parents to see their child's
feedback — no scores visible, comments only.

### Timed Sessions
Teacher sets a time limit. Session auto closes
when time expires. Students see countdown timer.

### Multi Subject Support
Extend beyond mathematics to science, English,
Urdu, and other subjects with subject specific
feedback dimensions and misconception libraries.

---

## Phase 8 — Deployment and Scale

### Production Infrastructure
- Render deployment with auto scaling
- Supabase PostgreSQL for production database
- CDN for static assets and uploaded files
- Monitoring and error alerting

### Custom Domain
Professional URL for school deployment:
explainly.school or similar

### Usage Analytics Dashboard
Admin view showing:
- Sessions created per week
- Average response rate
- Most common misconceptions across all schools
- Model accuracy metrics over time

### Multi School Support
Separate teacher accounts per school with
school level analytics for administrators.

---

## Key Learning from Pilot Testing

During the Atomcamp AI Bootcamp capstone pilot
the following specific issues were identified and
informed both prompt improvements and future work:
Issue 1 — Decomposition strategy misread as error
Fix applied: Added strategy recognition to prompt
Future: Two pass analysis will prevent this structurally
Issue 2 — Valid notation flagged incorrectly
Fix applied: Updated equals sign rule in prompt
Future: Model comparison testing to find better baseline
Issue 3 — Feedback too technical for students
Fix applied: Language rules added to prompt
Future: Interactive feedback loop for guided discovery
Issue 4 — Written only submissions penalised
Fix applied: Benefit of doubt instruction added
Future: Stronger model for written only submissions
Issue 5 — Roman Urdu vocabulary mishandled
Fix applied: Regional vocabulary added to prompt
Future: Aya-23 model evaluation for multilingual support