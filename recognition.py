import cv2
import pickle

# Load trained model and label map
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face_recognizer_model.xml")

with open("label_map.pkl", "rb") as f:
    label_map = pickle.load(f)

face_classifier = cv2.CascadeClassifier("C:/Users/Admin/Downloads/haarcascade_frontalface_default (5).xml")

def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None, None
    (x, y, w, h) = faces[0]
    return gray[y:y+h, x:x+w], (x, y, w, h)

cap = cv2.VideoCapture("rtsp://admin:Welcome%401234@192.168.2.80:554/Streaming/Channels/101")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    face, rect = detect_face(frame)

    if face is not None:
        face = cv2.resize(face, (200, 200))
        label_id, confidence = recognizer.predict(face)

        if confidence < 70:  # Lower is better
            name = label_map[label_id]
            label_text = f"{name} ({round(confidence, 2)})"
        else:
            label_text = "Unknown"

        (x, y, w, h) = rect
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) == 13:  # Enter key
        break

cap.release()
cv2.destroyAllWindows()
