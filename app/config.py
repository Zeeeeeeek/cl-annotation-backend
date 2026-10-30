import os

IMAGES_DIR = os.getenv("IMAGES_DIR", "data/images")
DATABASE_PATH = os.getenv("DATABASE_PATH", "data/annotations.db")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
PORT = int(os.getenv("PORT", "8080"))
DEVICE = os.getenv("DEVICE", "cuda")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))