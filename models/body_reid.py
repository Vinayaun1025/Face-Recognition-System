from torchreid.utils import FeatureExtractor

class BodyReID:
    def __init__(self, device):
        self.extractor = FeatureExtractor(
            model_name="osnet_x1_0",
            device=device
        )

    def get_embedding(self, img):
        return self.extractor(img)[0]
