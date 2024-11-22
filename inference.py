import torch
import cv2
from combined import CombinedModel
from dataset import transform_image  # Create a function to preprocess images

# Parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 5
model_path = "crime_detection_model.pth"

# Load Model
model = CombinedModel(num_classes=num_classes)
model.load_state_dict(torch.load(model_path))
model = model.to(device)
model.eval()

# Load Video
video_path = "test_video.mp4"
cap = cv2.VideoCapture(video_path)

frame_list = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    preprocessed_frame = transform_image(frame)
    frame_list.append(preprocessed_frame)

cap.release()

# Convert frame list to tensor
frames = torch.stack(frame_list).unsqueeze(0)  # Shape: (1, num_frames, 3, H, W)
frames = frames.to(device)

# Run inference
with torch.no_grad():
    outputs = model(frames)
    _, predicted = torch.max(outputs, 1)
    print(f"Predicted Class: {predicted.item()}")
