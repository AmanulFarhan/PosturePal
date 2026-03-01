import cv2
import asyncio
from collections import deque
from detection import extract_features_from_frame
from model import predict

class PostureEngine:

    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.prediction_buffer = deque(maxlen=6)

    async def start(self, websocket):

        while True:

            ret, frame = self.cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)

            features = extract_features_from_frame(frame)

            if features:

                prediction = predict(features)

                self.prediction_buffer.append(prediction)

                final_prediction = max(
                    set(self.prediction_buffer),
                    key=self.prediction_buffer.count
                )

                status = "GOOD" if final_prediction == 0 else "BAD"

                await websocket.send_json({
                    "status": status
                })

            await asyncio.sleep(0.1)