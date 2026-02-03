import os, cv2, numpy as np

def load_references(face_model, body_model, base_dir):
    face_refs, body_refs = [], []

    for f in os.listdir(f"{base_dir}/reference/face"):
        img = cv2.imread(f"{base_dir}/reference/face/{f}")
        emb = face_model.get_embedding(img)
        if emb is not None:
            face_refs.append(emb)

    for b in os.listdir(f"{base_dir}/reference/body"):
        img = cv2.imread(f"{base_dir}/reference/body/{b}")
        body_refs.append(body_model.get_embedding(img))

    return np.array(face_refs), np.array(body_refs)
