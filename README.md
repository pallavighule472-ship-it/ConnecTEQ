# ConnecTEQ — AI Hiring Agent

An end-to-end AI-powered recruitment pipeline built with LangGraph, FastAPI, AWS, and OpenAI. HR posts a job once — candidates apply via a shareable link — the pipeline automatically parses, scores, shortlists, and schedules interviews with zero manual effort.

---

## Architecture

```
Candidate applies (apply page)
        │
        ▼
POST /apply/{job_id}  ──►  S3 (resume stored)
        │
        ▼
LangGraph Pipeline (SQLite checkpointing)
   ├── 1. JD Structuring       — LLM parses raw JD into structured job profile
   ├── 2. Resume Parsing       — Extracts name, email, skills, experience from PDF/DOCX
   ├── 3. Match Scoring        — Scores candidate 0–10 across 4 weighted dimensions
   ├── 4. Shortlisting         — Auto-shortlists if score ≥ threshold, rejects otherwise
   ├── 5. Interview Scheduling — Books Google Calendar slot, sends invite email
   ├── ── [NodeInterrupt] ──   ← Pipeline pauses, waits for HR
   ├── 6. Transcript Analysis  — LLM analyzes interview transcript
   └── 7. Recommendation       — Generates strengths, concerns, hire recommendation
        │
        ▼
HR reviews in dashboard → Approve / Reject → Notification email sent
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Pipeline orchestration | LangGraph (StateGraph + SqliteSaver checkpointing) |
| LLM | OpenAI GPT-4o |
| Resume storage | AWS S3 |
| Async queue | AWS SQS |
| Email notifications | AWS SES / SMTP |
| Interview scheduling | Google Calendar API |
| API | FastAPI + Uvicorn |
| Frontend | Pure HTML/CSS/JS (no framework) |
| Database | SQLite (local) — swappable to PostgreSQL |

---

## Features

- **Job postings** — HR creates a job with JD, scoring weights, interview format, and max applicant limit
- **Auto-close** — Job posting closes automatically when applicant limit is reached
- **Shareable apply link** — `http://your-domain/apply?job=<job_id>` — candidates apply without HR involvement
- **Duplicate detection** — SHA256 hash prevents the same resume being processed twice
- **RBAC** — Admin (full access) vs Recruiter (view + interview actions only)
- **Audit log** — Every action logged with timestamp, role, and entity — Admin only
- **LinkedIn / Indeed webhooks** — `POST /webhook/linkedin` and `POST /webhook/indeed` for automated ingestion
- **Human-in-the-loop** — Pipeline pauses at interview stage using LangGraph `NodeInterrupt`

---

## Project Structure

```
├── HR_backend.py       # LangGraph pipeline — all nodes and graph definition
├── api.py              # FastAPI — all REST endpoints
├── index.html          # HR dashboard (single-file, no build step)
├── apply.html          # Public candidate apply page
├── worker.py           # SQS consumer worker
├── tests/
│   ├── conftest.py     # AWS mock setup
│   └── test_unit.py    # 22 unit tests
├── requirements.txt
└── .env                # See .env.example — never commit this
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/your-username/hiring-agent.git
cd hiring-agent
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Configure environment

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

```env
OPENAI_API_KEY=sk-...
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
S3_BUCKET_NAME=hiring-resume
SQS_QUEUE_URL=https://sqs.us-east-1.amazonaws.com/...
INTERVIEWER_EMAIL=hr@yourcompany.com
HR_EMAIL=hr@yourcompany.com
DEFAULT_MEETING_LINK=https://meet.google.com/...
ADMIN_API_KEY=your-admin-key
RECRUITER_API_KEY=your-recruiter-key
GOOGLE_SERVICE_ACCOUNT_JSON=path/to/service-account.json
```

### 3. Start the API server

```bash
uvicorn api:app --host 127.0.0.1 --port 8000
```

### 4. Open the dashboard

Open `index.html` in your browser (or navigate to `http://localhost:8000`).
Login with your `ADMIN_API_KEY`.

### 5. (Optional) Start the SQS worker

```bash
python worker.py
```

---

## API Endpoints

### Public
| Method | Endpoint | Description |
|---|---|---|
| GET | `/apply` | Serve candidate apply page |
| GET | `/jobs/{job_id}` | Get job details (used by apply page) |
| POST | `/apply/{job_id}` | Candidate submits resume |
| GET | `/health` | Health check |

### Authenticated (any role)
| Method | Endpoint | Description |
|---|---|---|
| GET | `/candidates` | List all candidates with pipeline summary |
| GET | `/candidate/{id}/status` | Get candidate's current pipeline stage |
| GET | `/candidate/{id}/resume` | Get S3 presigned URL for resume download |
| GET | `/jobs` | List all job postings |
| POST | `/interview/{id}/complete` | Mark interview as completed |
| POST | `/interview/{id}/transcript` | Upload interview transcript |
| POST | `/interview/{id}/cancel` | Cancel interview |

### Admin only
| Method | Endpoint | Description |
|---|---|---|
| POST | `/upload` | Manually submit a resume |
| POST | `/jobs` | Create a new job posting |
| DELETE | `/jobs/{id}` | Close a job posting |
| POST | `/candidate/{id}/decision` | Submit approve/reject decision |
| GET | `/audit-logs` | View full audit trail |

### Webhooks
| Method | Endpoint | Description |
|---|---|---|
| POST | `/webhook/linkedin` | LinkedIn application ingestion |
| POST | `/webhook/indeed` | Indeed application ingestion |

---

## Running Tests

```bash
pytest tests/ -v
```

22 unit tests covering: JSON parsing, regex extraction, scoring weights, shortlisting logic, routing, and evaluation dimensions. AWS is fully mocked — no real credentials needed.

---

## RBAC

| Permission | Admin | Recruiter |
|---|---|---|
| View candidates & status | ✅ | ✅ |
| Interview actions | ✅ | ✅ |
| Post / close jobs | ✅ | ❌ |
| Approve / reject candidates | ✅ | ❌ |
| View audit log | ✅ | ❌ |
