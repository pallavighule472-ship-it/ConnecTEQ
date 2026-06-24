import os
import uuid
import json
import logging
import secrets
import tempfile
import httpx
from datetime import datetime, timezone
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException, Depends
from fastapi.security import APIKeyHeader
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("api")

from fastapi.middleware.cors import CORSMiddleware
from HR_backend import ingest_resume, graph, s3, S3_BUCKET
try:
    from langgraph.errors import GraphInterrupt
except ImportError:
    GraphInterrupt = None


class _DB:
    """Normalises sqlite3 and psycopg3 into one interface."""
    def __init__(self):
        pg = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL")
        if pg:
            import psycopg
            self._c  = psycopg.connect(pg, autocommit=True)
            self._pg = True
        else:
            import sqlite3
            self._c  = sqlite3.connect(
                os.getenv("DATABASE_PATH", "hiring_agent.db"),
                check_same_thread=False,
            )
            self._pg = False

    def execute(self, sql: str, params=()):
        if self._pg:
            sql = sql.replace("?", "%s").replace(
                "INTEGER PRIMARY KEY AUTOINCREMENT", "SERIAL PRIMARY KEY"
            )
        return self._c.execute(sql, params)

    def commit(self):
        if not self._pg:
            self._c.commit()

    def executescript(self, sql: str):
        if self._pg:
            sql = sql.replace(
                "INTEGER PRIMARY KEY AUTOINCREMENT", "SERIAL PRIMARY KEY"
            )
            for stmt in (s.strip() for s in sql.split(";")):
                if stmt:
                    self._c.execute(stmt)
        else:
            self._c.executescript(sql)


_conn = _DB()


class CancelRequest(BaseModel):
    reason: str

class DecisionRequest(BaseModel):
    decision: str

class CreateJobRequest(BaseModel):
    title: str
    raw_jd: str
    skills_weight: float = 0.4
    experience_weight: float = 0.3
    education_weight: float = 0.2
    culture_fit_weight: float = 0.1
    interview_format: str = "video"
    max_applicants: int = 0  # 0 = unlimited

app = FastAPI()

# ════ TABLES ════
_conn.executescript("""
    CREATE TABLE IF NOT EXISTS jobs (
        job_id            TEXT PRIMARY KEY,
        title             TEXT NOT NULL,
        raw_jd            TEXT NOT NULL,
        skills_weight     REAL DEFAULT 0.4,
        experience_weight REAL DEFAULT 0.3,
        education_weight  REAL DEFAULT 0.2,
        culture_fit_weight REAL DEFAULT 0.1,
        interview_format  TEXT DEFAULT 'video',
        status            TEXT DEFAULT 'active',
        created_at        TEXT,
        applicant_count   INTEGER DEFAULT 0,
        max_applicants    INTEGER DEFAULT 0
    );
    CREATE TABLE IF NOT EXISTS audit_logs (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        ts          TEXT NOT NULL,
        actor_role  TEXT NOT NULL,
        action      TEXT NOT NULL,
        entity_type TEXT,
        entity_id   TEXT,
        detail      TEXT
    );
""")
_conn.commit()
try:
    _conn.execute("ALTER TABLE jobs ADD COLUMN max_applicants INTEGER DEFAULT 0")
    _conn.commit()
except Exception:
    pass  # column already exists

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ════ RBAC ════
# Two roles: admin (full access) and recruiter (view + interview actions only)
_ADMIN_KEY     = os.getenv("ADMIN_API_KEY") or os.getenv("API_KEY")
_RECRUITER_KEY = os.getenv("RECRUITER_API_KEY")

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def require_api_key(key: str = Depends(_api_key_header)) -> str:
    """Returns the caller's role ('admin' or 'recruiter') or raises 401."""
    if _ADMIN_KEY and key and secrets.compare_digest(key, _ADMIN_KEY):
        return "admin"
    if _RECRUITER_KEY and key and secrets.compare_digest(key, _RECRUITER_KEY):
        return "recruiter"
    logger.warning("Unauthorized — bad or missing API key")
    raise HTTPException(status_code=401, detail="Invalid or missing API key.")

def require_admin(role: str = Depends(require_api_key)) -> str:
    if role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required.")
    return role

# ════ AUDIT ════
def audit(action: str, entity_type: str, entity_id: str, role: str, detail: str = ""):
    try:
        _conn.execute(
            "INSERT INTO audit_logs (ts, actor_role, action, entity_type, entity_id, detail) VALUES (?,?,?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(), role, action, entity_type, entity_id, detail),
        )
        _conn.commit()
    except Exception:
        pass

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/debug")
def debug():
    """Diagnose the full pipeline — call this when candidates don't appear."""
    out = {}

    # ── 1. Database: jobs ────────────────────────────────────────────────────
    try:
        jobs = _conn.execute("SELECT job_id, title, status, applicant_count FROM jobs").fetchall()
        out["jobs"] = [
            {"job_id": r[0], "title": r[1], "status": r[2], "applicant_count": r[3]}
            for r in jobs
        ]
    except Exception as e:
        out["jobs"] = f"ERROR: {e}"

    # ── 2. Database: checkpoint count ───────────────────────────────────────
    try:
        rows = _conn.execute("SELECT COUNT(DISTINCT thread_id) FROM checkpoints").fetchone()
        out["checkpoint_threads"] = rows[0] if rows else 0
    except Exception as e:
        out["checkpoint_threads"] = f"ERROR (table may not exist yet): {e}"

    # ── 3. SQS: messages in queue ───────────────────────────────────────────
    try:
        import boto3 as _boto3
        _sqs = _boto3.client(
            "sqs",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION"),
        )
        attrs = _sqs.get_queue_attributes(
            QueueUrl=os.getenv("SQS_QUEUE_URL", ""),
            AttributeNames=["ApproximateNumberOfMessages",
                            "ApproximateNumberOfMessagesNotVisible"],
        )["Attributes"]
        out["sqs"] = {
            "visible_messages":   int(attrs.get("ApproximateNumberOfMessages", -1)),
            "in_flight_messages": int(attrs.get("ApproximateNumberOfMessagesNotVisible", -1)),
        }
    except Exception as e:
        out["sqs"] = f"ERROR: {e}"

    # ── 4. Env vars present (not values) ────────────────────────────────────
    required = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION",
                "S3_BUCKET_NAME", "SQS_QUEUE_URL", "OPENAI_API_KEY",
                "ADMIN_API_KEY", "HR_EMAIL"]
    out["env"] = {k: ("SET" if os.getenv(k) else "MISSING") for k in required}

    logger.info(f"[debug] {out}")
    return out


# ════ SERVE PAGES ════
@app.get("/")
def serve_dashboard():
    path = os.path.join(os.path.dirname(__file__), "index.html")
    return FileResponse(path, media_type="text/html")

@app.get("/apply")
def serve_apply_page():
    path = os.path.join(os.path.dirname(__file__), "apply.html")
    return FileResponse(path, media_type="text/html")


# ════ JOBS ════
def _row_to_job(r):
    max_ap = r[11] if len(r) > 11 else 0
    count  = r[10] or 0
    return {
        "job_id":            r[0],
        "title":             r[1],
        "raw_jd":            r[2],
        "skills_weight":     r[3],
        "experience_weight": r[4],
        "education_weight":  r[5],
        "culture_fit_weight":r[6],
        "interview_format":  r[7],
        "status":            r[8],
        "created_at":        r[9],
        "applicant_count":   count,
        "max_applicants":    max_ap,
        "slots_left":        max(0, max_ap - count) if max_ap > 0 else None,
        "is_full":           (max_ap > 0 and count >= max_ap),
    }

@app.post("/jobs")
def create_job(body: CreateJobRequest, role: str = Depends(require_admin)):
    job_id = str(uuid.uuid4())
    now    = datetime.now(timezone.utc).isoformat()
    _conn.execute(
        "INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (job_id, body.title, body.raw_jd,
         body.skills_weight, body.experience_weight,
         body.education_weight, body.culture_fit_weight,
         body.interview_format, "active", now, 0, body.max_applicants),
    )
    _conn.commit()
    limit_note = f"max {body.max_applicants}" if body.max_applicants else "unlimited"
    audit("job_created", "job", job_id, role, f"{body.title} ({limit_note})")
    logger.info(f"Job created — job_id={job_id}, title={body.title}, max={body.max_applicants}")
    return _row_to_job(_conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone())

@app.get("/jobs")
def list_jobs(_: str = Depends(require_api_key)):
    rows = _conn.execute("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()
    return {"jobs": [_row_to_job(r) for r in rows]}

@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    row = _conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Job not found.")
    return _row_to_job(row)

@app.delete("/jobs/{job_id}")
def close_job(job_id: str, role: str = Depends(require_admin)):
    _conn.execute("UPDATE jobs SET status='closed' WHERE job_id=?", (job_id,))
    _conn.commit()
    audit("job_closed", "job", job_id, role)
    return {"job_id": job_id, "status": "closed"}

@app.post("/apply/{job_id}")
async def apply_for_job(job_id: str, file: UploadFile = File(...)):
    row = _conn.execute(
        "SELECT * FROM jobs WHERE job_id=? AND status='active'", (job_id,)
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Job not found or no longer accepting applications.")

    max_ap = row[11] if len(row) > 11 else 0
    count  = row[10] or 0
    if max_ap > 0 and count >= max_ap:
        # race-condition safety: close and reject
        _conn.execute("UPDATE jobs SET status='closed' WHERE job_id=?", (job_id,))
        _conn.commit()
        raise HTTPException(status_code=409, detail="This job posting has reached its applicant limit and is now closed.")

    weights = {
        "skills":      row[3],
        "experience":  row[4],
        "education":   row[5],
        "culture_fit": row[6],
    }
    extension = os.path.splitext(file.filename)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        result = ingest_resume(tmp_path, row[2], weights, row[7], job_id=job_id)
    finally:
        os.remove(tmp_path)

    new_count = count + 1
    auto_closed = max_ap > 0 and new_count >= max_ap
    new_status  = "closed" if auto_closed else "active"

    _conn.execute(
        "UPDATE jobs SET applicant_count = ?, status = ? WHERE job_id=?",
        (new_count, new_status, job_id),
    )
    _conn.commit()

    if auto_closed:
        audit("job_auto_closed", "job", job_id, "system", f"reached limit of {max_ap}")
        logger.info(f"Job auto-closed — job_id={job_id}, limit={max_ap}")

    logger.info(f"Application received — job_id={job_id}, candidate_id={result['candidate_id']}, count={new_count}/{max_ap or 'unlimited'}")
    return {
        "status":       result.get("status", "queued"),
        "candidate_id": result["candidate_id"],
        "job_closed":   auto_closed,
    }

@app.post("/upload")
async def upload_resume(
    file: UploadFile = File(...),
    raw_jd: str = Form(...),
    skills_weight: float = Form(0.4),
    experience_weight: float = Form(0.3),
    education_weight: float = Form(0.2),
    culture_fit_weight: float = Form(0.1),
    interview_format: str = Form("video"),
    role: str = Depends(require_api_key),
):
    weights = {
        "skills": skills_weight,
        "experience": experience_weight,
        "education": education_weight,
        "culture_fit": culture_fit_weight,
    }
    extension = os.path.splitext(file.filename)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        result = ingest_resume(tmp_path, raw_jd, weights, interview_format)
        audit("resume_uploaded", "candidate", result["candidate_id"], role, file.filename)
        logger.info(f"Resume queued — candidate_id={result['candidate_id']}, format={interview_format}")
    finally:
        os.remove(tmp_path)
    return result

@app.get("/candidate/{candidate_id}/status")
def get_candidate_status(candidate_id: str, _: str = Depends(require_api_key)):
    config = {"configurable": {"thread_id": candidate_id}}
    try:
        snapshot = graph.get_state(config)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Candidate not found: {e}")

    if not snapshot or not snapshot.values:
        raise HTTPException(status_code=404, detail="Candidate not found.")

    state = snapshot.values
    next_nodes = snapshot.next

    candidate = state.get("candidate_profile")
    job      = state.get("job_profile")

    if state.get("decision") in ("approved", "rejected"):
        stage = f"completed — {state['decision']}"
    elif next_nodes:
        next_node = next_nodes[0]
        if next_node == "JD_structuring" and not job and not candidate:
            stage = "processing"
        else:
            stage = next_node
    else:
        stage = "completed"
    details  = state.get("interview_details") or {}

    logger.info(f"Status check — candidate_id={candidate_id}, stage={stage}")
    rec    = state.get("recommendation") or {}
    job_id = state.get("job_id") or ""
    job_row = _conn.execute("SELECT title FROM jobs WHERE job_id=?", (job_id,)).fetchone() if job_id else None
    return {
        "candidate_id":     candidate_id,
        "name":             candidate.name   if candidate else None,
        "email":            candidate.email  if candidate else None,
        "skills":           candidate.skills if candidate else [],
        "job_title":        job.job_title    if job       else None,
        "job_id":           job_id or None,
        "job_posting_title":job_row[0]       if job_row   else None,
        "stage":            stage,
        "match_score":      state.get("match_score"),
        "shortlisted":      state.get("shortlisted"),
        "interview_status": state.get("interview_status"),
        "scheduled_at":     details.get("scheduled_at"),
        "interview_format": state.get("interview_format"),
        "decision":         state.get("decision"),
        "recommendation": {
            "hire_recommendation":  rec.get("hire_recommendation"),
            "summary":              rec.get("summary"),
            "strengths":            rec.get("strengths", []),
            "concerns":             rec.get("concerns", []),
            "final_score":          rec.get("final_score"),
            "pre_interview_score":  rec.get("pre_interview_score"),
            "post_interview_score": rec.get("post_interview_score"),
            "confidence":           rec.get("confidence"),
            "flag_for_human":       rec.get("flag_for_human", False),
        },
    }


@app.get("/candidates")
def list_candidates(_: str = Depends(require_api_key)):
    try:
        rows = _conn.execute(
            "SELECT DISTINCT thread_id FROM checkpoints"
        ).fetchall()
    except Exception:
        return {"candidates": [], "summary": {"total": 0, "shortlisted": 0, "rejected": 0, "pending": 0}}

    candidates = []
    for (tid,) in rows:
        config = {"configurable": {"thread_id": tid}}
        try:
            snapshot = graph.get_state(config)
            if not snapshot or not snapshot.values:
                continue
            state      = snapshot.values
            next_nodes = snapshot.next

            candidate = state.get("candidate_profile")
            job       = state.get("job_profile")

            if state.get("decision") in ("approved", "rejected"):
                stage = f"completed — {state['decision']}"
            elif next_nodes:
                next_node = next_nodes[0]
                # Show "Processing..." when the pipeline hasn't started yet
                # (initial checkpoint: no job_profile and no candidate_profile)
                if next_node == "JD_structuring" and not job and not candidate:
                    stage = "processing"
                else:
                    stage = next_node
            else:
                stage = "completed"
            details   = state.get("interview_details") or {}

            job_id  = state.get("job_id") or ""
            job_row = _conn.execute(
                "SELECT title FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone() if job_id else None

            candidates.append({
                "candidate_id":      tid,
                "name":              candidate.name  if candidate else None,
                "email":             candidate.email if candidate else None,
                "job_title":         job.job_title   if job       else None,
                "job_id":            job_id or None,
                "job_posting_title": job_row[0] if job_row else None,
                "orphaned":          bool(job_id and not job_row),
                "match_score":       state.get("match_score"),
                "shortlisted":       state.get("shortlisted"),
                "interview_status":  state.get("interview_status"),
                "interview_format":  state.get("interview_format"),
                "scheduled_at":      details.get("scheduled_at"),
                "decision":          state.get("decision"),
                "stage":             stage,
            })
        except Exception:
            continue

    # pipeline funnel counts
    total      = len(candidates)
    shortlisted = sum(1 for c in candidates if c["shortlisted"])
    in_interview = sum(1 for c in candidates if c["interview_status"] == "scheduled")
    approved   = sum(1 for c in candidates if c["decision"] == "approved")
    rejected   = sum(1 for c in candidates if c["decision"] == "rejected")

    logger.info(f"Candidates list — total={total}")
    return {
        "candidates": candidates,
        "summary": {
            "total":        total,
            "shortlisted":  shortlisted,
            "in_interview": in_interview,
            "approved":     approved,
            "rejected":     rejected,
        },
    }


def _delete_checkpoint_rows(thread_id: str):
    """Delete all LangGraph checkpoint rows for a thread. Skips tables that don't exist."""
    for table in ("checkpoints", "checkpoint_blobs", "checkpoint_writes"):
        try:
            _conn.execute(f"DELETE FROM {table} WHERE thread_id=?", (thread_id,))
        except Exception:
            pass  # table may not exist in all LangGraph versions


@app.delete("/admin/candidates/{candidate_id}")
def delete_candidate(candidate_id: str, role: str = Depends(require_admin)):
    """Remove a single candidate's checkpoint rows (admin only)."""
    _delete_checkpoint_rows(candidate_id)
    _conn.commit()
    audit("candidate_deleted", "candidate", candidate_id, role)
    logger.info(f"Candidate deleted — candidate_id={candidate_id}")
    return {"deleted": candidate_id}


@app.delete("/admin/candidates")
def clear_orphaned_candidates(role: str = Depends(require_admin)):
    """Delete all candidate checkpoints whose job_id no longer exists in the jobs table."""
    try:
        rows = _conn.execute("SELECT DISTINCT thread_id FROM checkpoints").fetchall()
    except Exception:
        return {"deleted": 0, "ids": []}

    deleted = []
    for (tid,) in rows:
        config = {"configurable": {"thread_id": tid}}
        try:
            snapshot = graph.get_state(config)
            if snapshot and snapshot.values:
                job_id = snapshot.values.get("job_id") or ""
                if job_id:
                    exists = _conn.execute(
                        "SELECT 1 FROM jobs WHERE job_id=?", (job_id,)
                    ).fetchone()
                    if exists:
                        continue  # job still exists — keep this candidate
        except Exception:
            pass  # unreadable checkpoint — treat as orphaned

        _delete_checkpoint_rows(tid)
        deleted.append(tid)

    _conn.commit()
    logger.info(f"Orphaned candidates cleared — count={len(deleted)}, ids={deleted}")
    audit("orphaned_candidates_cleared", "system", "all", role, f"{len(deleted)} removed")
    return {"deleted": len(deleted), "ids": deleted}


@app.get("/candidate/{candidate_id}/resume")
def get_resume_url(candidate_id: str, _: str = Depends(require_api_key)):
    config = {"configurable": {"thread_id": candidate_id}}
    try:
        snapshot = graph.get_state(config)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Candidate not found: {e}")

    if not snapshot or not snapshot.values:
        raise HTTPException(status_code=404, detail="Candidate not found.")

    s3_key = snapshot.values.get("s3_key")
    if not s3_key:
        raise HTTPException(status_code=404, detail="Resume not found in pipeline state.")

    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET, "Key": s3_key},
        ExpiresIn=3600,
    )
    logger.info(f"Resume presigned URL generated — candidate_id={candidate_id}")
    return {"url": url, "expires_in": 3600}


@app.get("/candidate/{candidate_id}/interview-questions")
def get_interview_questions(candidate_id: str, _: str = Depends(require_api_key)):
    config = {"configurable": {"thread_id": candidate_id}}
    try:
        snapshot = graph.get_state(config)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Candidate not found: {e}")
    if not snapshot or not snapshot.values:
        raise HTTPException(status_code=404, detail="Candidate not found.")

    state     = snapshot.values
    candidate = state.get("candidate_profile")
    job       = state.get("job_profile")
    if not candidate or not job:
        raise HTTPException(status_code=400, detail="Pipeline has not completed profiling yet.")

    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage

    llm    = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)
    prompt = f"""You are preparing an interviewer for a hiring conversation.

CANDIDATE
Name      : {candidate.name}
Skills    : {', '.join(candidate.skills or [])}
Experience: {candidate.experience or 'Not specified'}
Education : {candidate.education or 'Not specified'}

JOB
Title     : {job.job_title}
Required  : {', '.join((job.skills.must_have if job.skills else []) or [])}
Experience: {job.experience_required.raw if job.experience_required else 'Not specified'}

Match Score: {state.get('match_score', 'N/A')}/10

Generate exactly 9 interview questions in 3 categories.
Focus especially on gaps — skills the job requires that are weak or absent in the candidate profile.

Return ONLY valid JSON, no markdown fences:
{{
  "technical": [
    {{"q": "question text", "why": "one-line reason this matters for the role"}}
  ],
  "gap_probing": [
    {{"q": "question text", "why": "one-line reason — what gap this probes"}}
  ],
  "behavioral": [
    {{"q": "question text", "why": "one-line reason — what trait this reveals"}}
  ]
}}

3 questions per category. Gap probing must target the weakest or missing areas specifically."""

    try:
        raw  = llm.invoke([SystemMessage(content="Return only valid JSON."), HumanMessage(content=prompt)])
        text = raw.content.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        questions = json.loads(text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate questions: {e}")

    audit("questions_generated", "candidate", candidate_id, "system")
    logger.info(f"Interview questions generated — candidate_id={candidate_id}")
    return {"candidate_id": candidate_id, "questions": questions}


def _resume_graph(state_update: dict, candidate_id: str) -> str:
    """Resume an interrupted graph with a state update.
    update_state merges the values into the checkpoint; invoke(None) then
    continues from where the graph paused rather than restarting from START.
    Returns 'paused' or 'completed'. Raises HTTPException on real errors.
    """
    config = {"configurable": {"thread_id": candidate_id}}
    try:
        graph.update_state(config, state_update)
        graph.invoke(None, config=config)
        return "completed"
    except Exception as e:
        if GraphInterrupt and isinstance(e, GraphInterrupt):
            return "paused"  # pipeline paused at next human checkpoint — expected
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/interview/{candidate_id}/complete")
def mark_interview_complete(candidate_id: str, role: str = Depends(require_api_key)):
    status = _resume_graph({"interview_status": "completed"}, candidate_id)
    audit("interview_completed", "candidate", candidate_id, role)
    return {"status": status, "candidate_id": candidate_id}


@app.post("/interview/{candidate_id}/transcript")
async def upload_transcript(candidate_id: str, file: UploadFile = File(...), role: str = Depends(require_api_key)):
    extension = os.path.splitext(file.filename)[1].lower() or ".txt"
    s3_key = f"transcripts/{candidate_id}{extension}"
    content = await file.read()
    s3.put_object(Bucket=S3_BUCKET, Key=s3_key, Body=content)
    status = _resume_graph({"interview_status": "completed", "transcript_s3_key": s3_key}, candidate_id)
    audit("transcript_uploaded", "candidate", candidate_id, role, s3_key)
    return {"status": status, "candidate_id": candidate_id, "transcript_s3_key": s3_key}


@app.post("/interview/{candidate_id}/cancel")
def cancel_interview(candidate_id: str, body: CancelRequest, role: str = Depends(require_api_key)):
    status = _resume_graph({"cancel_reason": body.reason}, candidate_id)
    audit("interview_cancelled", "candidate", candidate_id, role, body.reason)
    return {"status": status, "candidate_id": candidate_id}


@app.post("/candidate/{candidate_id}/decision")
def submit_decision(candidate_id: str, body: DecisionRequest, role: str = Depends(require_admin)):
    if body.decision not in ("approved", "rejected"):
        raise HTTPException(status_code=400, detail="decision must be 'approved' or 'rejected'")
    _resume_graph({"decision": body.decision}, candidate_id)
    audit("decision_submitted", "candidate", candidate_id, role, body.decision)
    return {"status": "decided", "candidate_id": candidate_id, "decision": body.decision}


@app.get("/audit-logs")
def get_audit_logs(limit: int = 100, _: str = Depends(require_admin)):
    rows = _conn.execute(
        "SELECT id, ts, actor_role, action, entity_type, entity_id, detail FROM audit_logs ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return {"logs": [
        {"id": r[0], "ts": r[1], "actor_role": r[2], "action": r[3],
         "entity_type": r[4], "entity_id": r[5], "detail": r[6]}
        for r in rows
    ]}


# ════ ANALYTICS ════
@app.get("/analytics")
def get_analytics(_: str = Depends(require_api_key)):
    try:
        rows = _conn.execute("SELECT DISTINCT thread_id FROM checkpoints").fetchall()
    except Exception:
        rows = []

    scores, skills_counter, formats, decisions = [], {}, {}, {"approved": 0, "rejected": 0, "pending": 0}
    total = shortlisted = interviewed = approved = rejected = 0

    for (tid,) in rows:
        config = {"configurable": {"thread_id": tid}}
        try:
            snap = graph.get_state(config)
            if not snap or not snap.values:
                continue
            st  = snap.values
            total += 1
            dec = st.get("decision")
            sc  = st.get("match_score")
            fmt = st.get("interview_format", "unknown")

            if sc is not None:
                scores.append(sc)
            formats[fmt] = formats.get(fmt, 0) + 1

            if st.get("shortlisted"):
                shortlisted += 1
            if st.get("interview_status") in ("scheduled", "completed"):
                interviewed += 1
            if dec == "approved":
                approved += 1
                decisions["approved"] += 1
            elif dec == "rejected":
                rejected += 1
                decisions["rejected"] += 1
            else:
                decisions["pending"] += 1

            candidate = st.get("candidate_profile")
            if candidate and candidate.skills:
                for skill in candidate.skills:
                    s = skill.strip().lower()
                    if s:
                        skills_counter[s] = skills_counter.get(s, 0) + 1
        except Exception:
            continue

    avg_score = round(sum(scores) / len(scores), 2) if scores else None
    score_dist = {"0-3": 0, "4-6": 0, "7-8": 0, "9-10": 0}
    for s in scores:
        if s <= 3:   score_dist["0-3"]  += 1
        elif s <= 6: score_dist["4-6"]  += 1
        elif s <= 8: score_dist["7-8"]  += 1
        else:        score_dist["9-10"] += 1

    top_skills = sorted(skills_counter.items(), key=lambda x: x[1], reverse=True)[:10]

    shortlist_rate     = round(shortlisted / total * 100)   if total       else 0
    interview_rate     = round(interviewed / shortlisted * 100) if shortlisted else 0
    hire_rate          = round(approved    / interviewed * 100) if interviewed else 0

    jobs_row = _conn.execute("SELECT COUNT(*), SUM(applicant_count) FROM jobs WHERE status='active'").fetchone()

    return {
        "funnel": {
            "applied":      total,
            "shortlisted":  shortlisted,
            "interviewed":  interviewed,
            "approved":     approved,
            "rejected":     rejected,
        },
        "rates": {
            "shortlist_rate": shortlist_rate,
            "interview_rate": interview_rate,
            "hire_rate":      hire_rate,
        },
        "avg_score":    avg_score,
        "score_dist":   score_dist,
        "top_skills":   [{"skill": k, "count": v} for k, v in top_skills],
        "formats":      [{"format": k, "count": v} for k, v in formats.items()],
        "decisions":    decisions,
        "active_jobs":  jobs_row[0] if jobs_row else 0,
    }


# ════ BIAS CHECK ════
@app.get("/candidate/{candidate_id}/bias-check")
def bias_check(candidate_id: str, _: str = Depends(require_api_key)):
    config = {"configurable": {"thread_id": candidate_id}}
    try:
        snap = graph.get_state(config)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
    if not snap or not snap.values:
        raise HTTPException(status_code=404, detail="Candidate not found.")

    state     = snap.values
    candidate = state.get("candidate_profile")
    job       = state.get("job_profile")
    if not candidate:
        raise HTTPException(status_code=400, detail="Pipeline has not completed profiling yet.")

    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage

    llm    = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    prompt = f"""You are a fairness auditor reviewing an AI hiring decision for potential bias.

CANDIDATE
Name      : {candidate.name}
Skills    : {', '.join(candidate.skills or [])}
Experience: {candidate.experience or 'Not specified'}
Education : {candidate.education or 'Not specified'}

JOB TITLE : {job.job_title if job else 'Unknown'}
AI SCORE  : {state.get('match_score', 'N/A')}/10
SHORTLISTED: {state.get('shortlisted')}

Analyse this hiring record for demographic signals that may have influenced the AI score.
Check for:
1. Gender signals (name implies gender)
2. Ethnicity/nationality signals (name or institution implies background)
3. Age signals (graduation year, years of experience imply age)
4. Socioeconomic signals (institution prestige, unpaid internships)

Return ONLY valid JSON:
{{
  "risk_level": "low|medium|high",
  "signals": [
    {{"type": "gender|ethnicity|age|socioeconomic", "detail": "specific observation", "recommendation": "what HR should do"}}
  ],
  "overall_assessment": "2-3 sentence summary",
  "anonymization_suggestions": ["field 1 to anonymize", "field 2 to anonymize"]
}}

If no signals found, return risk_level "low" with empty signals array."""

    try:
        raw  = llm.invoke([SystemMessage(content="Return only valid JSON."), HumanMessage(content=prompt)])
        text = raw.content.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        result = json.loads(text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bias check failed: {e}")

    audit("bias_check", "candidate", candidate_id, "system", result.get("risk_level", ""))
    logger.info(f"Bias check — candidate_id={candidate_id}, risk={result.get('risk_level')}")
    return {"candidate_id": candidate_id, "bias": result}


@app.post("/webhook/linkedin")
async def linkedin_webhook(request: Request):
    payload = await request.json()

    resume_url = payload.get("resumeUrl")
    raw_jd = payload.get("jobDescription", "")

    if not resume_url:
        raise HTTPException(status_code=400, detail="No resume URL in payload")

    async with httpx.AsyncClient() as client:
        response = await client.get(resume_url)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name
    try:
        result = ingest_resume(tmp_path, raw_jd)
    finally:
        os.remove(tmp_path)

    return {"status": "queued", "candidate_id": result["candidate_id"]}

@app.post("/webhook/indeed")
async def indeed_webhook(request: Request):
    payload = await request.json()

    resume_url = payload.get("resume", {}).get("url")
    raw_jd = payload.get("job", {}).get("description", "")

    if not resume_url:
        raise HTTPException(status_code=400, detail="No resume URL in payload")

    async with httpx.AsyncClient() as client:
        response = await client.get(resume_url)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name
    try:
        result = ingest_resume(tmp_path, raw_jd)
    finally:
        os.remove(tmp_path)

    return {"status": "queued", "candidate_id": result["candidate_id"]}
