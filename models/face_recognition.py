import insightface

class FaceRecognizer:
    def __init__(self):
        self.app = insightface.app.FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider"]
        )
        self.app.prepare(ctx_id=0)

    def get_embedding(self, img):
        faces = self.app.get(img)
        return faces[0].embedding if faces else None
