import torch

BASE_DIR = "cctv_prod"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FPS_SKIP = 3
DETECTION_CONF = 0.4
MATCH_THRESHOLD = 0.75
