import os
import cv2

STREAM_URL = os.getenv("STREAM_URL")

if not STREAM_URL:
    raise ValueError("STREAM_URL env var not set")

cap = cv2.VideoCapture(STREAM_URL)

ret, frame = cap.read()

print("Stream reachable:", ret)

if ret:
    print("Frame shape:", frame.shape)

cap.release()
