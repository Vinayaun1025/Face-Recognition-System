from sklearn.metrics.pairwise import cosine_similarity

def is_missing_person(face_emb, body_emb, face_refs, body_refs, threshold):
    body_score = max(cosine_similarity([body_emb], body_refs))

    if body_score < 0.6:
        return False, body_score

    face_score = 0
    if face_emb is not None:
        face_score = max(cosine_similarity([face_emb], face_refs))

    final = 0.5 * face_score + 0.5 * body_score
    return final > threshold, final
