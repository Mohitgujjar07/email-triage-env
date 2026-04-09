import asyncio
import json
import websockets
from models import EmailAction, EmailObservation, EmailState, TriageAction


class EmailTriageEnv:
    """
    OpenEnv-compatible client for the Email Triage environment.

    Usage (async):
        async with EmailTriageEnv(base_url="ws://localhost:8000") as env:
            obs, state = await env.reset()
            result = await env.step(EmailAction(action=TriageAction.REPLY))

    Usage (sync):
        with EmailTriageEnv(base_url="ws://localhost:8000").sync() as env:
            obs, state = env.reset()
    """

    def __init__(self, base_url: str = "ws://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self._ws = None

    async def __aenter__(self):
        ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://")
        self._ws = await websockets.connect(f"{ws_url}/ws")
        return self

    async def __aexit__(self, *args):
        if self._ws:
            await self._ws.close()

    async def reset(self) -> tuple[EmailObservation, EmailState]:
        await self._ws.send(json.dumps({"method": "reset"}))
        raw = await self._ws.recv()
        data = json.loads(raw)
        return EmailObservation(**data["observation"]), EmailState(**data["state"])

    async def step(self, action: EmailAction):
        await self._ws.send(json.dumps({"method": "step", "action": action.model_dump()}))
        raw = await self._ws.recv()
        data = json.loads(raw)
        return (
            EmailObservation(**data["observation"]),
            data["reward"],
            data["done"],
            EmailState(**data["state"]),
        )

    async def state(self) -> EmailState:
        await self._ws.send(json.dumps({"method": "state"}))
        raw = await self._ws.recv()
        data = json.loads(raw)
        return EmailState(**data["state"])

    def sync(self):
        return _SyncWrapper(self)


class _SyncWrapper:
    def __init__(self, async_env: EmailTriageEnv):
        self._env = async_env
        self._loop = asyncio.new_event_loop()

    def __enter__(self):
        self._loop.run_until_complete(self._env.__aenter__())
        return self

    def __exit__(self, *args):
        self._loop.run_until_complete(self._env.__aexit__(*args))
        self._loop.close()

    def reset(self):
        return self._loop.run_until_complete(self._env.reset())

    def step(self, action: EmailAction):
        return self._loop.run_until_complete(self._env.step(action))

    def state(self):
        return self._loop.run_until_complete(self._env.state())


# Quick test / demo
if __name__ == "__main__":
    async def demo():
        print("Connecting to Email Triage environment...")
        async with EmailTriageEnv(base_url="ws://localhost:8000") as env:
            obs, state = await env.reset()
            print(f"\nEpisode started! {state.total_emails} emails to triage.\n")

            while not state.done:
                print(f"--- Email {state.step_count + 1}/{state.total_emails} ---")
                print(f"From:    {obs.sender}")
                print(f"Subject: {obs.subject}")
                print(f"Body:    {obs.body[:80]}...")
                print()

                # Simple heuristic agent for demo
                subject_lower = obs.subject.lower()
                if any(w in subject_lower for w in ["urgent", "critical", "down", "bug", "legal"]):
                    action = TriageAction.ESCALATE
                elif any(w in subject_lower for w in ["schedule", "call", "feedback", "question"]):
                    action = TriageAction.REPLY
                elif any(w in subject_lower for w in ["invoice", "expire", "reminder"]):
                    action = TriageAction.SNOOZE
                else:
                    action = TriageAction.ARCHIVE

                obs, reward, done, state = await env.step(EmailAction(action=action))
                print(f"Agent chose: {action.value}  |  Reward: {reward:+.1f}  |  Score so far: {state.score:.1f}")
                if obs.feedback:
                    print(f"Feedback: {obs.feedback}")
                print()

            print(f"Episode complete! Final score: {state.score:.1f}/{state.total_emails}")

    asyncio.run(demo())
