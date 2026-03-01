from fastapi import FastAPI, WebSocket
from engine import PostureEngine

app = FastAPI()
engine = PostureEngine()

@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await engine.start(websocket)