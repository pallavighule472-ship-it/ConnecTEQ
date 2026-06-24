import os
import json
import logging
import tempfile
import boto3
from dotenv import load_dotenv
load_dotenv()

from HR_backend import graph, s3

try:
    from langgraph.errors import GraphInterrupt
except ImportError:
    GraphInterrupt = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("worker")

S3_BUCKET = os.getenv("S3_BUCKET_NAME")
SQS_QUEUE_URL = os.getenv("SQS_QUEUE_URL")

sqs = boto3.client(
    "sqs",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION")
)

def process_message(message: dict):
    body = json.loads(message["Body"])
    candidate_id = body["candidate_id"]
    s3_key = body["s3_key"]
    raw_jd = body["raw_jd"]

    logger.info(f"[{candidate_id}] processing started — s3_key={s3_key}")

    extension = os.path.splitext(s3_key)[1]
    tmp_path = os.path.join(tempfile.gettempdir(), f"{candidate_id}{extension}")

    s3.download_file(S3_BUCKET, s3_key, tmp_path)

    try:
        initial_state = {
            "raw_jd":           raw_jd,
            "candidate_id":     candidate_id,
            "file_path":        tmp_path,
            "s3_key":           s3_key,
            "scoring_weights":  body.get("scoring_weights", {}),
            "interview_format": body.get("interview_format", "video"),
            "job_id":           body.get("job_id", ""),
        }
        config = {"configurable": {"thread_id": candidate_id}}
        result = graph.invoke(initial_state, config=config)
        logger.info(f"[{candidate_id}] processing completed — decision={result.get('decision')}")
        return result
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _delete_message(message: dict):
    sqs.delete_message(
        QueueUrl=SQS_QUEUE_URL,
        ReceiptHandle=message["ReceiptHandle"],
    )


def run_worker():
    logger.info("Worker started. Polling SQS...")
    poll_count = 0
    while True:
        response = sqs.receive_message(
            QueueUrl=SQS_QUEUE_URL,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=20,
        )

        messages = response.get("Messages", [])
        if not messages:
            poll_count += 1
            if poll_count % 5 == 0:          # log every ~100 s of silence
                logger.info(f"Worker alive — polled {poll_count} times, no messages yet")
            continue
        poll_count = 0

        for message in messages:
            mid = message.get("MessageId", "unknown")
            try:
                result = process_message(message)
                logger.info(f"[{result.get('candidate_id')}] done — deleting SQS message")
                _delete_message(message)

            except Exception as e:
                if GraphInterrupt and isinstance(e, GraphInterrupt):
                    # Pipeline paused for human-in-the-loop (e.g. waiting for interview).
                    # State is saved in the checkpointer — safe to delete the SQS message.
                    logger.info(f"[{mid}] pipeline paused (awaiting human action) — state saved, deleting SQS message")
                    _delete_message(message)
                else:
                    logger.error(f"[{mid}] processing failed — {e}", exc_info=True)
                    # Message NOT deleted — SQS auto-retries after visibility timeout


if __name__ == "__main__":
    run_worker()
