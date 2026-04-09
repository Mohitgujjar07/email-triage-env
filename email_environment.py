import uuid
import random
from typing import Optional
from models import EmailAction, EmailObservation, EmailState, TriageAction


EMAILS = [
    {
        "subject": "URGENT: Server is down in production",
        "body": "Our main API server has been returning 500 errors for the last 10 minutes. Customers are affected. Need immediate help.",
        "sender": "devops@company.com",
        "correct_action": TriageAction.ESCALATE,
        "feedback": "Production outages always need immediate escalation.",
    },
    {
        "subject": "Meeting notes from last Tuesday",
        "body": "Hi, please find the notes from our weekly sync attached. No action items for you.",
        "sender": "alice@company.com",
        "correct_action": TriageAction.ARCHIVE,
        "feedback": "FYI emails with no action items should be archived.",
    },
    {
        "subject": "Can we schedule a call this week?",
        "body": "Hi, I wanted to discuss the Q3 proposal with you. Are you free Thursday or Friday afternoon?",
        "sender": "partner@client.com",
        "correct_action": TriageAction.REPLY,
        "feedback": "Scheduling requests need a reply to confirm availability.",
    },
    {
        "subject": "Invoice #4521 due in 30 days",
        "body": "Your invoice of $2,400 is due on the 15th next month. No immediate action required.",
        "sender": "billing@vendor.com",
        "correct_action": TriageAction.SNOOZE,
        "feedback": "Future-dated invoices should be snoozed until closer to the due date.",
    },
    {
        "subject": "Your password will expire in 3 days",
        "body": "Please reset your company password before it expires to avoid being locked out.",
        "sender": "it@company.com",
        "correct_action": TriageAction.SNOOZE,
        "feedback": "Reminder emails with future deadlines are best snoozed.",
    },
    {
        "subject": "Legal: Contract review needed ASAP",
        "body": "The NDA with Acme Corp needs to be reviewed and signed by EOD. Please escalate to legal if you haven't already.",
        "sender": "ceo@company.com",
        "correct_action": TriageAction.ESCALATE,
        "feedback": "Legal deadlines from senior staff require immediate escalation.",
    },
    {
        "subject": "Re: your question about pricing",
        "body": "Thanks for reaching out! Our enterprise plan starts at $500/month. Let me know if you'd like a demo.",
        "sender": "sales@softwaretool.com",
        "correct_action": TriageAction.ARCHIVE,
        "feedback": "Unsolicited sales replies with no follow-up needed should be archived.",
    },
    {
        "subject": "Feedback on your presentation",
        "body": "Hi, I really enjoyed your talk yesterday. I had a few thoughts I wanted to share — would love to connect.",
        "sender": "colleague@partner.org",
        "correct_action": TriageAction.REPLY,
        "feedback": "Positive outreach deserves a reply to maintain relationships.",
    },
    {
        "subject": "Critical bug in payment flow",
        "body": "We found a bug where payments are being double-charged. This is happening live. Please escalate immediately.",
        "sender": "qa@company.com",
        "correct_action": TriageAction.ESCALATE,
        "feedback": "Any live bug affecting payments is a critical escalation.",
    },
    {
        "subject": "Newsletter: AI trends in 2026",
        "body": "This week in AI: LLMs get smaller, RL environments go open source, and more. Unsubscribe below.",
        "sender": "news@aiweekly.com",
        "correct_action": TriageAction.ARCHIVE,
        "feedback": "Newsletters with no required action should be archived.",
    },
]


class EmailEnvironment:
    def __init__(self):
        self.episode_id: Optional[str] = None
        self.step_count: int = 0
        self.score: float = 0.0
        self.queue: list = []
        self.current_email: Optional[dict] = None
        self.done: bool = False

    def reset(self) -> tuple[EmailObservation, EmailState]:
        self.episode_id = str(uuid.uuid4())
        self.step_count = 0
        self.score = 0.0
        self.done = False
        self.queue = random.sample(EMAILS, min(5, len(EMAILS)))
        self.current_email = self.queue[self.step_count]

        obs = EmailObservation(
            email_id=f"email_{self.step_count}",
            subject=self.current_email["subject"],
            body=self.current_email["body"],
            sender=self.current_email["sender"],
        )
        state = EmailState(
            episode_id=self.episode_id,
            step_count=self.step_count,
            total_emails=len(self.queue),
            score=self.score,
            done=self.done,
        )
        return obs, state

    def step(self, action: EmailAction) -> tuple[EmailObservation, float, bool, EmailState]:
        correct = self.current_email["correct_action"]
        feedback = self.current_email["feedback"]

        reward = 1.0 if action.action == correct else -0.5
        self.score += reward
        self.step_count += 1
        self.done = self.step_count >= len(self.queue)

        obs = EmailObservation(
            email_id=f"email_{self.step_count - 1}",
            subject=self.current_email["subject"],
            body=self.current_email["body"],
            sender=self.current_email["sender"],
            correct_action=correct,
            feedback=feedback,
        )

        if not self.done:
            self.current_email = self.queue[self.step_count]
            next_obs = EmailObservation(
                email_id=f"email_{self.step_count}",
                subject=self.current_email["subject"],
                body=self.current_email["body"],
                sender=self.current_email["sender"],
            )
        else:
            next_obs = obs

        state = EmailState(
            episode_id=self.episode_id,
            step_count=self.step_count,
            total_emails=len(self.queue),
            score=self.score,
            done=self.done,
        )
        return next_obs, reward, self.done, state

    def get_state(self) -> EmailState:
        return EmailState(
            episode_id=self.episode_id or "",
            step_count=self.step_count,
            total_emails=len(self.queue),
            score=self.score,
            done=self.done,
        )
