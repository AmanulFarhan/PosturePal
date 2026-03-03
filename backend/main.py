from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from engine import PostureEngine

app = FastAPI()

@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    engine = PostureEngine()

    try:
        await engine.start(websocket)
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print("Error:", e)
    finally:
        engine.stop()