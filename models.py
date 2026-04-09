from enum import Enum
from typing import Optional
from pydantic import BaseModel


class TriageAction(str, Enum):
    REPLY = "reply"
    ARCHIVE = "archive"
    ESCALATE = "escalate"
    SNOOZE = "snooze"


class EmailAction(BaseModel):
    action: TriageAction
    reason: Optional[str] = None


class EmailObservation(BaseModel):
    email_id: str
    subject: str
    body: str
    sender: str
    correct_action: Optional[TriageAction] = None  # revealed after step
    feedback: Optional[str] = None


class EmailState(BaseModel):
    episode_id: str
    step_count: int
    total_emails: int
    score: float
    done: bool
