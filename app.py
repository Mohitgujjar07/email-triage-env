import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import json
import traceback

from models import EmailAction, EmailObservation, EmailState
from server.email_environment import EmailEnvironment

app = FastAPI(title="Email Triage OpenEnv")

env = EmailEnvironment()


@app.get("/")
async def root():
    return {"status": "ok", "env": "email_triage", "version": "1.0.0"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            method = data.get("method")

            if method == "reset":
                obs, state = env.reset()
                await websocket.send_text(json.dumps({
                    "observation": obs.model_dump(),
                    "state": state.model_dump(),
                }))

            elif method == "step":
                action = EmailAction(**data["action"])
                next_obs, reward, done, state = env.step(action)
                await websocket.send_text(json.dumps({
                    "observation": next_obs.model_dump(),
                    "reward": reward,
                    "done": done,
                    "state": state.model_dump(),
                }))

            elif method == "state":
                state = env.get_state()
                await websocket.send_text(json.dumps({
                    "state": state.model_dump(),
                }))

            else:
                await websocket.send_text(json.dumps({"error": f"Unknown method: {method}"}))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_text(json.dumps({"error": str(e), "trace": traceback.format_exc()}))


@app.get("/web", response_class=HTMLResponse)
async def web_interface():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web_ui.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())
