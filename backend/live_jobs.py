"""
Thread-safe in-memory job manager for FastAPI demo workloads.

This module is intentionally small and dependency-free so it can be imported
directly by the API layer without changing the rest of the codebase.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4


JOB_STATUS_PENDING = "pending"
JOB_STATUS_RUNNING = "running"
JOB_STATUS_SUCCEEDED = "succeeded"
JOB_STATUS_FAILED = "failed"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    """Best-effort JSON-safe conversion for common Python objects."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    return str(value)


@dataclass
class JobRecord:
    job_id: str
    kind: str
    status: str = JOB_STATUS_PENDING
    stage: str = "queued"
    progress: float = 0.0
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    started_at: str | None = None
    finished_at: str | None = None
    input_payload: dict[str, Any] = field(default_factory=dict)
    latest_payload: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)

    def snapshot(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "status": self.status,
            "stage": self.stage,
            "progress": float(self.progress),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "input_payload": _json_safe(self.input_payload),
            "latest_payload": _json_safe(self.latest_payload),
            "result": _json_safe(self.result),
            "error": self.error,
            "metadata": _json_safe(self.metadata),
            "history": [_json_safe(item) for item in self.history],
        }


class JobNotFoundError(KeyError):
    """Raised when a job id is not present in the manager."""


class LiveJobManager:
    """Small in-memory job manager with polling-friendly snapshots."""

    def __init__(self, *, max_history: int = 128, storage_dir: str | Path | None = None) -> None:
        self._lock = RLock()
        self._jobs: dict[str, JobRecord] = {}
        self._max_history = max(8, int(max_history))
        self._storage_dir = Path(storage_dir or (Path(__file__).resolve().parents[1] / "results" / "live_runs"))
        self._storage_dir.mkdir(parents=True, exist_ok=True)

    def create_job(
        self,
        kind: str = "optimization",
        *,
        input_payload: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            job_id = uuid4().hex
            record = JobRecord(
                job_id=job_id,
                kind=str(kind),
                input_payload=dict(input_payload or {}),
                latest_payload=dict(input_payload or {}),
                metadata=dict(metadata or {}),
            )
            self._attach_artifact_metadata(record)
            self._jobs[job_id] = record
            self._append_history(record, event="created", message="Job created", stage=record.stage)
            self._persist_record(record)
            return record.snapshot()

    def update_job(
        self,
        job_id: str,
        *,
        stage: str | None = None,
        progress: float | None = None,
        payload: dict[str, Any] | None = None,
        message: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            record = self._require(job_id)
            if stage is not None:
                record.stage = str(stage)
                if record.status == JOB_STATUS_PENDING:
                    record.status = JOB_STATUS_RUNNING
                    if record.started_at is None:
                        record.started_at = _utc_now()
            if progress is not None:
                record.progress = float(max(0.0, min(1.0, progress)))
            if payload is not None:
                record.latest_payload = dict(payload)
            if metadata:
                record.metadata.update(metadata)
            record.updated_at = _utc_now()
            self._append_history(
                record,
                event="update",
                message=message or "Job updated",
                stage=record.stage,
                progress=record.progress,
            )
            self._persist_record(record)
            return record.snapshot()

    def mark_running(self, job_id: str, *, stage: str = "running", message: str | None = None) -> dict[str, Any]:
        return self.update_job(job_id, stage=stage, progress=0.01 if self.get(job_id)["progress"] == 0.0 else None, message=message or "Job running")

    def mark_progress(
        self,
        job_id: str,
        *,
        progress: float,
        stage: str | None = None,
        payload: dict[str, Any] | None = None,
        message: str | None = None,
    ) -> dict[str, Any]:
        return self.update_job(job_id, stage=stage, progress=progress, payload=payload, message=message)

    def mark_complete(
        self,
        job_id: str,
        *,
        result: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
        message: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            record = self._require(job_id)
            if payload is not None:
                record.latest_payload = dict(payload)
            record.result = dict(result or payload or record.latest_payload)
            record.status = JOB_STATUS_SUCCEEDED
            record.stage = "completed"
            record.progress = 1.0
            record.finished_at = _utc_now()
            record.updated_at = record.finished_at
            self._append_history(record, event="completed", message=message or "Job completed", stage=record.stage, progress=record.progress)
            self._persist_record(record)
            return record.snapshot()

    def mark_failed(
        self,
        job_id: str,
        *,
        error: str | Exception,
        payload: dict[str, Any] | None = None,
        message: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            record = self._require(job_id)
            if payload is not None:
                record.latest_payload = dict(payload)
            record.status = JOB_STATUS_FAILED
            record.stage = "failed"
            record.progress = min(record.progress, 1.0)
            record.error = str(error)
            record.finished_at = _utc_now()
            record.updated_at = record.finished_at
            self._append_history(record, event="failed", message=message or "Job failed", stage=record.stage, progress=record.progress, error=record.error)
            self._persist_record(record)
            return record.snapshot()

    def get(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            return self._require(job_id).snapshot()

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            return [record.snapshot() for record in self._jobs.values()]

    def delete(self, job_id: str) -> bool:
        with self._lock:
            return self._jobs.pop(job_id, None) is not None

    def clear(self) -> None:
        with self._lock:
            self._jobs.clear()

    def _require(self, job_id: str) -> JobRecord:
        try:
            return self._jobs[str(job_id)]
        except KeyError as exc:
            raise JobNotFoundError(job_id) from exc

    def _attach_artifact_metadata(self, record: JobRecord) -> None:
        run_dir = self._storage_dir / record.job_id
        checkpoints_dir = run_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        record.metadata.setdefault(
            "artifacts",
            {
                "run_dir": str(run_dir),
                "snapshot_file": str(run_dir / "snapshot.json"),
                "history_file": str(run_dir / "history.jsonl"),
                "latest_payload_file": str(run_dir / "latest_payload.json"),
                "result_file": str(run_dir / "result.json"),
                "checkpoints_dir": str(checkpoints_dir),
            },
        )

    def _write_json(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")

    def _append_jsonl(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_json_safe(payload)) + "\n")

    def _persist_record(self, record: JobRecord) -> None:
        self._attach_artifact_metadata(record)
        artifacts = record.metadata["artifacts"]
        self._write_json(Path(artifacts["snapshot_file"]), record.snapshot())
        self._write_json(Path(artifacts["latest_payload_file"]), record.latest_payload)
        if record.result is not None:
            self._write_json(Path(artifacts["result_file"]), record.result)

    def _persist_checkpoint(self, record: JobRecord, event_entry: dict[str, Any]) -> str:
        self._attach_artifact_metadata(record)
        checkpoints_dir = Path(record.metadata["artifacts"]["checkpoints_dir"])
        stage = str(event_entry.get("stage", "stage")).replace(":", "_").replace(" ", "_")
        checkpoint_path = checkpoints_dir / f"{len(record.history) + 1:04d}_{stage}.json"
        checkpoint_payload = {
            "event": event_entry,
            "status": record.status,
            "stage": record.stage,
            "progress": record.progress,
            "latest_payload": record.latest_payload,
        }
        self._write_json(checkpoint_path, checkpoint_payload)
        return str(checkpoint_path)

    def _append_history(self, record: JobRecord, **event: Any) -> None:
        event_entry = {"timestamp": _utc_now(), **{key: _json_safe(value) for key, value in event.items()}}
        event_entry["checkpoint_file"] = self._persist_checkpoint(record, event_entry)
        record.history.append(event_entry)
        self._append_jsonl(Path(record.metadata["artifacts"]["history_file"]), event_entry)
        if len(record.history) > self._max_history:
            record.history[:] = record.history[-self._max_history :]


JOB_MANAGER = LiveJobManager()


__all__ = [
    "JOB_STATUS_PENDING",
    "JOB_STATUS_RUNNING",
    "JOB_STATUS_SUCCEEDED",
    "JOB_STATUS_FAILED",
    "JobRecord",
    "JobNotFoundError",
    "LiveJobManager",
    "JOB_MANAGER",
]
