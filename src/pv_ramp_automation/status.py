from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class SectionStatus:
    section_name: str
    started_at: str
    finished_at: str
    status: str
    message: str
    key_metrics: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None


def write_status(path: str | Path, status: SectionStatus | dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(status) if isinstance(status, SectionStatus) else status
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def section_status(section_name: str, started_at: str, status: str, message: str, key_metrics: dict[str, Any] | None = None, error_message: str | None = None) -> SectionStatus:
    return SectionStatus(
        section_name=section_name,
        started_at=started_at,
        finished_at=utc_now(),
        status=status,
        message=message,
        key_metrics=key_metrics or {},
        error_message=error_message,
    )
