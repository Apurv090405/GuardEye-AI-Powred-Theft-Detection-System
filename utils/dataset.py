import os
import torch
from torch.utils.data import Dataset
from config import DATA_SOURCE, RANDOM_STATE, TRAIN_TEST_SPLIT
from utils.preprocess import process_video

class CrimeDataset(Dataset):
    def __init__(self, train=True):
        torch.manual_seed(RANDOM_STATE)
        self.data = []
        self.labels = []
        inclusion_prob = TRAIN_TEST_SPLIT if train else 1 - TRAIN_TEST_SPLIT

        print(f"Loading {'train' if train else 'test'} dataset...")
        for label, path in DATA_SOURCE.items():
            print(f"Processing label: {label}")
            for file in os.listdir(path):
                if file.endswith('.mp4') and torch.rand(1).item() <= inclusion_prob:
                    video_path = os.path.join(path, file)
                    frames = process_video(video_path)
                    self.data.extend(frames)
                    self.labels.extend([label] * len(frames))
        print(f"Loaded {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], torch.tensor(self.label_str2id(self.labels[idx]))

    @staticmethod
    def label_str2id(label):
        return list(DATA_SOURCE.keys()).index(label)

    @staticmethod
    def label_id2str(label_id):
        return list(DATA_SOURCE.keys())[label_id]
