import cv2
import os
import numpy as np

def prepare_training_data(data_folder_path):
    faces = []
    labels = []
    label_map = {}
    label_id = 0

    for person_name in os.listdir(data_folder_path):
        person_path = os.path.join(data_folder_path, person_name)
        if not os.path.isdir(person_path):
            continue

        label_map[label_id] = person_name  # For mapping later

        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            faces.append(image)
            labels.append(label_id)

        label_id += 1

    return faces, np.array(labels), label_map

faces, labels, label_map = prepare_training_data("C:/Users/Admin/OneDrive - Unite Cloud services LLP/Desktop/dataset")

# Initialize and train the recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, labels)

# Save model and label map
recognizer.write("face_recognizer_model.xml")

# Save label map
import pickle
with open("label_map.pkl", "wb") as f:
    pickle.dump(label_map, f)

print("Training complete. Model and label map saved.")

