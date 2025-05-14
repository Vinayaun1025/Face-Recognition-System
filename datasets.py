import cv2
import os

# Load the face classifier
face_classifier = cv2.CascadeClassifier("C:/Users/Admin/Downloads/haarcascade_frontalface_default (5).xml")

def face_extractor(img):
    if img is None or img.size == 0:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None

    for (x, y, w, h) in faces:
        cropped_face = img[y:y+h, x:x+w]
        return cropped_face

# Ask for user name or ID
person_name = input("Enter person's name or ID: ").strip()

# Create folder if it doesn't exist
dataset_path = "C:/Users/Admin/OneDrive - Unite Cloud services LLP/Desktop/dataset"
person_folder = os.path.join(dataset_path, person_name)
os.makedirs(person_folder, exist_ok=True)

cap = cv2.VideoCapture("rtsp://admin:Welcome%401234@192.168.2.80:554/Streaming/Channels/101")
count = 0

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Failed to capture frame. Exiting...")
        break

    face = face_extractor(frame)

    if face is not None:
        count += 1
        face = cv2.resize(face, (200, 200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        filename = os.path.join(person_folder, f"{count}.jpg")
        cv2.imwrite(filename, face)

        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Face Cropper", face)
    else:
        print("Face not found.")
        pass

    # Exit when Enter key is pressed or 100 images are saved
    if cv2.waitKey(1) == 13 or count == 100:
        break

cap.release()
cv2.destroyAllWindows()
print(f"Dataset collection completed for {person_name}.")
