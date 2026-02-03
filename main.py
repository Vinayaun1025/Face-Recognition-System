import os, cv2
from config import *
from models.detector import PersonDetector
from models.face_recognition import FaceRecognizer
from models.body_reid import BodyReID
from tracking.tracker import Tracker
from utils.embeddings import load_references
from utils.fusion import is_missing_person
from utils.video_utils import should_process

detector = PersonDetector()
face_model = FaceRecognizer()
body_model = BodyReID(DEVICE)
tracker = Tracker()

face_refs, body_refs = load_references(face_model, body_model, BASE_DIR)

videos = os.listdir(f"{BASE_DIR}/videos")

for cam_id, video in enumerate(videos, 1):
    cap = cv2.VideoCapture(f"{BASE_DIR}/videos/{video}")
    out = cv2.VideoWriter(
        f"{BASE_DIR}/outputs/videos/cam_{cam_id}.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        10,
        (int(cap.get(3)), int(cap.get(4)))
    )

    frame_id = 0
    track_cache = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        if not should_process(frame_id, FPS_SKIP):
            continue

        boxes = detector.detect(frame)
        tracks = tracker.update(boxes, frame)

        for t in tracks:
            if not t.is_confirmed():
                continue

            tid = t.track_id
            x1,y1,x2,y2 = map(int, t.to_ltrb())
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            if tid not in track_cache:
                face_emb = face_model.get_embedding(crop)
                body_emb = body_model.get_embedding(crop)
                track_cache[tid] = (face_emb, body_emb)
            else:
                face_emb, body_emb = track_cache[tid]

            match, score = is_missing_person(
                face_emb, body_emb, face_refs, body_refs, MATCH_THRESHOLD
            )

            if match:
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),3)
                cv2.putText(frame,f"MISSING {score:.2f}",
                            (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

        out.write(frame)

    cap.release()
    out.release()
