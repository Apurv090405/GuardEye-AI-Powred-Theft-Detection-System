import cv2
import torch
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from config import RESOLUTION, FRAME_INTERVAL

# Transformation Pipeline
transformer = Compose([
    ToTensor(),
    Normalize(mean=[0.5], std=[0.5]),
    Resize((RESOLUTION, RESOLUTION)),
])

def process_video(path):
    """
    Process a video file into frames and apply transformations.
    """
    if not path or not isinstance(path, str):
        return [], []

    cap = cv2.VideoCapture(path)
    frames = []
    success, frame = cap.read()
    while success:
        try:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            processed_frame = transformer(gray_frame)
            frames.append(processed_frame)
        except Exception as e:
            print(f"Error processing frame: {e}")
        # Skip frames for intervals
        for _ in range(FRAME_INTERVAL):
            success, frame = cap.read()
    return frames
