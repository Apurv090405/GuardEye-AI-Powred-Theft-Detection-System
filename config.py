import torch

# Configuration
RESOLUTION = 512
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_SOURCE = {
    "Robbery": "/kaggle/input/anomalydetectiondatasetucf/Anomaly-Videos-Part-3/Anomaly-Videos-Part-3/Robbery",
    "Shoplifting": "/kaggle/input/anomalydetectiondatasetucf/Anomaly-Videos-Part-4/Anomaly-Videos-Part-4/Shoplifting",
    "Stealing": "/kaggle/input/anomalydetectiondatasetucf/Anomaly-Videos-Part-4/Anomaly-Videos-Part-4/Stealing",
    "Burglary": "/kaggle/input/anomalydetectiondatasetucf/Anomaly-Videos-Part_2/Anomaly-Videos-Part-2/Burglary",
    "Normal_Videos": "/kaggle/input/anomalydetectiondatasetucf/Normal_Videos_for_Event_Recognition/Normal_Videos_for_Event_Recognition",
}
TRAIN_TEST_SPLIT = 0.80
FRAME_INTERVAL = 30
RANDOM_STATE = 42
BATCH_SIZE = 512
NUM_EPOCHS = 50
LEARNING_RATE = 0.01
