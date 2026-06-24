"""
In-memory background-job tracker for training runs.

A bake-off across 5 models takes a couple of minutes on CPU, so training is
triggered as a background task and the client polls for status. A single
process-wide dict is sufficient at this project's scale (one container, no
multi-worker scaling); swap for Celery/RQ + Redis if that ever changes.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Literal

JobStatus = Literal["pending", "running", "completed", "failed"]


@dataclass
class Job:
    id: str
    status: JobStatus = "pending"
    message: str = "Queued"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    result: dict[str, Any] | None = None
    error: str | None = None


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = Lock()

    def create(self) -> Job:
        job = Job(id=str(uuid.uuid4()))
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            for k, v in fields.items():
                setattr(job, k, v)
            job.updated_at = time.time()


job_store = JobStore()
